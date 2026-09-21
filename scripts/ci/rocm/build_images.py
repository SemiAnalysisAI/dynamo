#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the rendered runtime and standard test Dockerfiles inside Slurm."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm.docker_daemon import DockerDaemon
from slurm_common import atomic_json, read_json, sha256_file
from slurm_verify import require


def run(command, log, **kwargs):
    """Keep each build stage's complete output and propagate its real exit code."""
    with log.open("w") as output:
        output.write(json.dumps([str(arg) for arg in command]) + "\n")
        output.flush()
        subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, check=True, **kwargs
        )


def inspect_image(docker, tag, destination):
    content = subprocess.check_output(
        docker.command("image", "inspect", tag), text=True
    )
    inspected = json.loads(content)
    require(len(inspected) == 1, "Expected exactly one Docker image")
    atomic_json(destination, inspected[0])
    return inspected[0]


def record_image(docker, image_id, results, scratch):
    # A rootful daemon cannot traverse a root-squashed NFS home directory.
    # Bind node-local inputs and copy evidence back as the submitting user.
    inspection = scratch / "image-inspection"
    inspection.mkdir(mode=0o700)
    (inspection / "home").mkdir(mode=0o700)
    shutil.copytree(results / "controller", inspection / "controller")
    for name in ("request.json", "image-build.json"):
        shutil.copyfile(results / name, inspection / name)
    # The Slurm UID need not have an image passwd entry. Libraries still need
    # a username and writable home/cache directories during import.
    try:
        run(
            docker.command(
                "run",
                "--rm",
                "--network=none",
                "--cgroup-parent",
                docker.cgroup_parent,
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--env",
                "HOME=/results/home",
                "--env",
                "USER=dynamo",
                "--env",
                "LOGNAME=dynamo",
                "--env",
                "XDG_CACHE_HOME=/results/home/.cache",
                "--env",
                "TORCHINDUCTOR_CACHE_DIR=/results/home/.cache/torchinductor",
                "--mount",
                f"type=bind,src={inspection},dst=/results",
                "--mount",
                f"type=bind,src={inspection / 'controller'},dst=/results/controller,readonly",
                "--entrypoint",
                "/opt/dynamo/venv/bin/python3",
                image_id,
                "-I",
                "/results/controller/rocm/record_image.py",
                "--results",
                "/results",
            ),
            results / "record-image.log",
        )
    finally:
        # Preserve partial diagnostics if image inspection fails.
        for name in (
            "installed-packages.log",
            "native-packages.log",
            "native-linkage.log",
            "pip-check.log",
            "build-spec.json",
            "build-manifest.json",
        ):
            output = inspection / name
            if output.is_file() and not output.is_symlink():
                shutil.copyfile(output, results / name)


def build_images(source, results, scratch):
    require(os.environ.get("SLURM_JOB_ID"), "Image builds require a Slurm allocation")
    request = read_json(results / "request.json")
    contract = read_json(Path(__file__).with_name("contract.json"))
    renderer = scratch / "renderer"
    run([sys.executable, "-m", "venv", str(renderer)], results / "renderer-venv.log")
    python = renderer / "bin/python3"
    run(
        [str(python), "-m", "pip", "install", "Jinja2==3.1.6", "PyYAML==6.0.3"],
        results / "renderer-dependencies.log",
    )
    run(
        [
            str(python),
            "container/render.py",
            "--device",
            "rocm",
            "--framework",
            "vllm",
            "--target",
            "runtime",
            "--platform",
            "linux/amd64",
            "--output-short-filename",
        ],
        results / "render-runtime.log",
        cwd=source,
    )
    runtime_dockerfile = source / "container/rendered.Dockerfile"
    test_dockerfile = source / "container/Dockerfile.test"
    shutil.copyfile(runtime_dockerfile, results / "rendered-runtime.Dockerfile")
    shutil.copyfile(test_dockerfile, results / "test-image.Dockerfile")
    runtime_tag = "dynamo-rocm-runtime:" + request["run_key"]
    test_tag = "dynamo-rocm-test:" + request["run_key"]
    with DockerDaemon(scratch, results / "docker-daemon.log") as docker:
        run(
            docker.command(
                "buildx",
                "build",
                "--load",
                "--progress=plain",
                "--platform=linux/amd64",
                "--cgroup-parent",
                docker.cgroup_parent,
                "--network=host",
                "--tag",
                runtime_tag,
                "--build-arg",
                "DYNAMO_COMMIT_SHA=" + request["source_sha"],
                "--file",
                str(runtime_dockerfile),
                str(source),
            ),
            results / "build-runtime.log",
        )
        runtime = inspect_image(
            docker, runtime_tag, results / "runtime-image-inspect.json"
        )
        expected_base = contract["base_uri"].removeprefix(
            "docker://registry-1.docker.io#"
        )
        require(
            runtime["Config"]["Labels"].get("org.opencontainers.image.base.name")
            == expected_base,
            "Rendered runtime uses a different base image than the qualification contract",
        )
        for name, arguments in {
            "sanity_check": [
                "/workspace/dev/sanity_check.py",
                "--runtime-check",
                "--no-gpu-check",
            ],
            "pip_check": ["-m", "pip", "check"],
        }.items():
            run(
                docker.command(
                    "run",
                    "--rm",
                    "--network=none",
                    "--cgroup-parent",
                    docker.cgroup_parent,
                    "--entrypoint",
                    "/opt/dynamo/venv/bin/python3",
                    runtime["Id"],
                    *arguments,
                ),
                results / ("runtime-" + name + ".log"),
            )
        run(
            docker.command(
                "buildx",
                "build",
                "--load",
                "--progress=plain",
                "--platform=linux/amd64",
                "--cgroup-parent",
                docker.cgroup_parent,
                "--network=host",
                "--target=test_image",
                "--tag",
                test_tag,
                "--build-arg",
                "BASE_IMAGE=" + runtime_tag,
                "--file",
                str(test_dockerfile),
                str(source),
            ),
            results / "build-test.log",
        )
        test = inspect_image(docker, test_tag, results / "test-image-inspect.json")
        atomic_json(
            results / "image-build.json",
            {
                "platform": "linux/amd64",
                "runtime_dockerfile_sha256": sha256_file(runtime_dockerfile),
                "test_dockerfile_sha256": sha256_file(test_dockerfile),
                "runtime_image_id": runtime["Id"],
                "test_image_id": test["Id"],
                "runtime_sanity": {
                    "status": "passed",
                    "image_id": runtime["Id"],
                    "checks": {"sanity_check": 0, "pip_check": 0},
                },
            },
        )
        # Inspect the final test image without installing or copying anything into it.
        record_image(docker, test["Id"], results, scratch)
        # Enroot's dockerd importer saves this exact image from the private daemon.
        # No registry push or replacement image is involved.
        enroot_env = {
            **docker.env,
            "ENROOT_CACHE_PATH": str(scratch / "enroot-cache"),
            "ENROOT_DATA_PATH": str(scratch / "enroot-data"),
            "ENROOT_RUNTIME_PATH": str(scratch / "enroot-runtime"),
            "ENROOT_MAX_PROCESSORS": os.environ["SLURM_CPUS_PER_TASK"],
        }
        for name in ("ENROOT_CACHE_PATH", "ENROOT_DATA_PATH", "ENROOT_RUNTIME_PATH"):
            Path(enroot_env[name]).mkdir(mode=0o700)
        run(
            [
                "enroot",
                "import",
                "--output",
                str(results / "dynamo-test.building.sqsh"),
                "dockerd://" + test_tag,
            ],
            results / "export-test-image.log",
            env=enroot_env,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, terminate)
    build_images(args.source.resolve(), args.results.resolve(), args.scratch.resolve())


if __name__ == "__main__":
    main()
