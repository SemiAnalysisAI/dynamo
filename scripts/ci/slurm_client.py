#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the ROCm Slurm qualification from a GitHub Actions workflow dispatch."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from slurm_common import (
    atomic_json,
    controller_manifest,
    read_json,
    sha256_file,
    sha256_json,
    validate_run_key,
    validate_sha,
)
from slurm_verify import verify_collected

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SSH_HOST = "ci-slurm-login"
RUN_TIMEOUT = 17100
FINALIZE_TIMEOUT = 240
PYTHON = [
    "env",
    "-u",
    "PYTHONPATH",
    "-u",
    "PYTHONHOME",
    "-u",
    "PYTHONUSERBASE",
    "-u",
    "PYTHONSTARTUP",
    "PYTHONNOUSERSITE=1",
    "python3",
    "-B",
]


@dataclass(frozen=True)
class ActionsContext:
    workspace: Path
    runner_temp: Path
    sha: str
    run_key: str
    expected_uid: int
    preferred_partition: str
    fallback_partition: str

    @property
    def ssh_directory(self):
        return self.runner_temp / "dynamo-rocm" / "ssh"

    @property
    def output(self):
        return self.runner_temp / "dynamo-rocm" / "evidence"

    @classmethod
    def from_environment(cls):
        if (
            os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or os.environ.get("GITHUB_REF_TYPE") != "branch"
        ):
            raise ValueError(
                "This driver requires a GitHub Actions workflow_dispatch on a branch"
            )
        paths = {}
        for name in ("GITHUB_WORKSPACE", "RUNNER_TEMP"):
            path = Path(os.environ[name])
            if not path.is_absolute() or not path.is_dir():
                raise ValueError(f"{name} must be an existing absolute directory")
            paths[name] = path.resolve()
        workspace = paths["GITHUB_WORKSPACE"]
        runner_temp = paths["RUNNER_TEMP"]
        if workspace != REPO:
            raise ValueError("The driver must run from the GITHUB_WORKSPACE checkout")
        if runner_temp.is_relative_to(workspace) or workspace.is_relative_to(
            runner_temp
        ):
            raise ValueError("RUNNER_TEMP must be separate from GITHUB_WORKSPACE")
        sha = os.environ["GITHUB_SHA"]
        validate_sha(sha)
        if os.environ.get("GITHUB_WORKFLOW_SHA") != sha:
            raise ValueError(
                "The workflow and checkout must use the same GitHub commit"
            )
        for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "SLURM_UID"):
            if not re.fullmatch(r"[1-9][0-9]*", os.environ[name]):
                raise ValueError(f"{name} must be a positive integer")
        run_key = f"gh-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
        validate_run_key(run_key)
        preferred = os.environ.get("PREFERRED_PARTITION", "compute-1")
        fallback = os.environ.get("FALLBACK_PARTITION", "compute-0")
        for name in (preferred, fallback):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
                raise ValueError("Invalid partition name")
        if "auto" in (preferred, fallback) or preferred == fallback:
            raise ValueError("Preferred and fallback partitions must be distinct names")
        return cls(
            workspace,
            runner_temp,
            sha,
            run_key,
            int(os.environ["SLURM_UID"]),
            preferred,
            fallback,
        )


def run_checked(command, *, timeout=60, **kwargs):
    return subprocess.run(command, check=True, timeout=timeout, **kwargs)


def git(path, *args):
    return run_checked(["git", "-C", str(path), *args], capture_output=True).stdout


def clean_checkout(path, sha):
    validate_sha(sha)
    if git(path, "rev-parse", "HEAD").decode().strip() != sha:
        raise ValueError(f"Checkout HEAD does not match the requested SHA: {path}")
    if git(path, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError(f"Checkout must be clean, including untracked files: {path}")


def source_archive(source, destination, sha):
    """Archive tracked, materialized checkout bytes with deterministic metadata."""
    clean_checkout(source, sha)
    entries = []
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for raw in sorted(git(source, "ls-files", "--stage", "-z").split(b"\0")):
            if not raw:
                continue
            header, name = raw.split(b"\t", 1)
            git_mode, _, stage = header.decode().split()
            name = name.decode("utf-8")
            path = source / name
            if stage != "0" or git_mode not in ("100644", "100755", "120000"):
                raise ValueError(f"Unresolved entry or unsupported submodule: {name}")
            if name.startswith("/") or ".." in PurePosixPath(name).parts:
                raise ValueError("Unsafe source path")
            info = tarfile.TarInfo(name)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            if git_mode == "120000":
                target = os.readlink(path)
                data = target.encode()
                info.type = tarfile.SYMTYPE
                info.linkname = target
                info.mode = 0o777
                archive.addfile(info)
                entry = {"type": "symlink", "linkname": target}
            else:
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Unexpected tracked file type: {name}")
                with path.open("rb") as stream:
                    if stream.read(256).startswith(
                        b"version https://git-lfs.github.com/spec/v1\n"
                    ):
                        raise ValueError(
                            f"Unmaterialized LFS pointer: {name}; run git lfs pull"
                        )
                    stream.seek(0)
                    info.size = path.stat().st_size
                    info.mode = 0o755 if git_mode == "100755" else 0o644
                    archive.addfile(info, stream)
                data = None
                entry = {"type": "file"}
            entry.update(
                path=name,
                size=len(data) if data is not None else info.size,
                mode=info.mode,
                sha256=hashlib.sha256(data).hexdigest()
                if data is not None
                else sha256_file(path),
            )
            entries.append(entry)
    clean_checkout(source, sha)
    return {
        "schema_version": 1,
        "source_sha": sha,
        "archive_sha256": sha256_file(destination),
        "files": entries,
    }


def stage_controller(destination):
    destination.mkdir()
    for name in git(REPO, "ls-files", "-z", "scripts/ci").decode().split("\0"):
        if not name:
            continue
        relative = Path(name).relative_to("scripts/ci")
        if "tests" in relative.parts or relative.suffix == ".md":
            continue
        source = REPO / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Unsupported controller entry: {name}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = controller_manifest(destination)
    atomic_json(destination / "bundle-manifest.json", manifest)
    return sha256_json(manifest)


class Connection:
    def __init__(self, context):
        self.prefix = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "Compression=yes",
            "-F",
            str(context.ssh_directory / "config"),
            SSH_HOST,
        ]
        self.deadline = None

    def call(self, words, *, timeout=60, **kwargs):
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Controller operation deadline exhausted")
        return run_checked(
            [*self.prefix, shlex.join([str(w) for w in words])],
            timeout=timeout,
            **kwargs,
        )

    def python(self, code, *args, timeout=60):
        response = self.call(
            [*PYTHON, "-c", code, *args], timeout=timeout, capture_output=True
        )
        return json.loads(response.stdout)


def remote_program():
    """Send the same reviewed bootstrap helper used by offline tests."""
    return (HERE / "slurm_remote.py").read_text()


def preflight(connection, context):
    return connection.python(remote_program(), "preflight", str(context.expected_uid))


def collect(connection, remote_run, output, timeout=120):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        connection.call(
            [*PYTHON, "-c", remote_program(), "collect", remote_run],
            timeout=timeout,
            stdout=archive,
            stderr=subprocess.PIPE,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as handle:
            for member in handle.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not member.isfile():
                    raise ValueError("Unsafe evidence archive")
                target = output / member.name
                if any(parent.is_symlink() for parent in [target, *target.parents]):
                    raise ValueError("Evidence destination contains a symlink")
            handle.extractall(output, filter="data")


def remote_action(connection, remote_run, action, deadline=240):
    words = [
        *PYTHON,
        str(PurePosixPath(remote_run) / "controller/slurm-submit.py"),
        action,
        "--run-dir",
        remote_run,
        "--deadline-seconds",
        str(deadline),
    ]
    return json.loads(
        connection.call(
            words, timeout=deadline if action == "finalize" else 60, capture_output=True
        ).stdout
    )


def verdict(output):
    request = read_json(output / "request.json")
    intended = output / "expected-request.json"
    if request != read_json(intended):
        raise ValueError("Collected request differs from the original Actions request")
    receipt = read_json(output / "receipt.json")
    terminal = read_json(output / "terminal.json")
    waiter = read_json(output / "wait-result.json")
    controller = read_json(output / "controller-result.json")
    workload = read_json(output / "completed.json")
    expected = str(receipt["job_id"])
    if not re.fullmatch(r"[1-9][0-9]*", expected):
        raise ValueError("Invalid scheduler job ID")
    if (
        receipt.get("run_key") != request["run_key"]
        or receipt.get("source_sha") != request["source_sha"]
        or terminal.get("run_key") != request["run_key"]
    ):
        raise ValueError("Scheduler receipt belongs to a different run")
    if (
        controller.get("status") != "terminal"
        or terminal.get("state") != "COMPLETED"
        or terminal.get("exit_code") != "0:0"
        or terminal.get("restarts") != 0
        or str(terminal.get("job_id")) != expected
        or waiter.get("returncode") != 0
        or waiter.get("missing_record") is not False
        or str(waiter.get("job_id")) != expected
        or workload.get("status") != "passed"
        or str(workload.get("job_id")) != expected
    ):
        raise ValueError(
            "Scheduler/waiter/workload evidence does not establish success"
        )
    for key in (
        "run_key",
        "source_sha",
        "archive_sha256",
        "controller_sha",
        "controller_bundle_sha256",
    ):
        if workload.get(key) != request[key]:
            raise ValueError(f"Workload verdict identity mismatch: {key}")
    image = read_json(output / "image-manifest.json")
    validate_sha(image["sqsh_sha256"], 64)
    if workload.get("sqsh_sha256") != image["sqsh_sha256"]:
        raise ValueError("Workload image identity mismatch")
    verify_collected(request, output)
    return {"status": "passed", "job_id": expected, "run_key": request["run_key"]}


def wait_for_run(connection, remote_run, output, timeout):
    deadline = time.monotonic() + timeout
    if connection.deadline is not None:
        deadline = min(deadline, connection.deadline)
    # One deadline bounds retries, sleeps, status calls, and final collection.
    connection.deadline = deadline
    previous = None
    last_collection = time.monotonic()
    while time.monotonic() < deadline:
        try:
            state = remote_action(connection, remote_run, "status")
        except (OSError, subprocess.SubprocessError) as error:
            atomic_json(
                output / "connection-error.json",
                {"error": str(error), "at": time.time()},
            )
            time.sleep(min(20, max(0, deadline - time.monotonic())))
            continue
        atomic_json(output / "client-status.json", state)
        display = (state["phase"], state.get("receipt", {}).get("job_id"))
        if display != previous:
            print(json.dumps(state), flush=True)
            previous = display
        if not state.get("active"):
            collect(connection, remote_run, output)
            result = verdict(output)
            atomic_json(output / "client-result.json", result)
            print(json.dumps(result), flush=True)
            return
        if time.monotonic() - last_collection >= 120:
            try:
                collect(connection, remote_run, output, timeout=45)
            except (OSError, subprocess.SubprocessError) as error:
                atomic_json(
                    output / "collection-error.json",
                    {"error": str(error), "at": time.time()},
                )
            last_collection = time.monotonic()
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        "Client deadline reached; the Actions finalizer must collect evidence"
    )


def start_run(context, connection, info):
    source = context.workspace
    output = context.output
    clean_checkout(source, context.sha)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "client.json").exists():
        raise ValueError("This Actions attempt already started a Slurm run")
    (output / "run-key.txt").write_text(context.run_key + "\n")
    remote_run = str(PurePosixPath(info["root"]) / "runs" / context.run_key)
    client_state = {
        "run_key": context.run_key,
        "remote_run": remote_run,
        "ssh_host": SSH_HOST,
        "source_sha": context.sha,
        "started_at": time.time(),
        "deadline_at": time.time() + max(0, connection.deadline - time.monotonic()),
    }
    print(f"Run {context.run_key}; evidence {output}; remote {remote_run}", flush=True)
    with tempfile.TemporaryDirectory(
        prefix="slurm-source-", dir=context.runner_temp
    ) as scratch:
        stage = Path(scratch) / "run"
        stage.mkdir()
        source_manifest = source_archive(source, stage / "source.tar", context.sha)
        atomic_json(stage / "source-manifest.json", source_manifest)
        bundle_sha = stage_controller(stage / "controller")
        clean_checkout(source, context.sha)
        request = {
            "schema_version": 1,
            "run_key": context.run_key,
            "source_sha": context.sha,
            "archive_sha256": source_manifest["archive_sha256"],
            "controller_sha": context.sha,
            "controller_bundle_sha256": bundle_sha,
            "suite": "aggregate",
            "preferred_partition": context.preferred_partition,
            "fallback_partition": context.fallback_partition,
            "gpus": 1,
            "cpus": 16,
            "mem_gib": 64,
            "time_limit_minutes": 240,
            "queue_timeout_seconds": 1800,
            "controller_timeout_seconds": RUN_TIMEOUT,
            "expected_uid": context.expected_uid,
        }
        atomic_json(stage / "request.json", request)
        atomic_json(output / "request.json", request)
        atomic_json(output / "expected-request.json", request)
        package = Path(scratch) / "package.tar"
        with tarfile.open(package, "w") as archive:
            for path in sorted(stage.rglob("*")):
                archive.add(path, arcname=path.relative_to(stage), recursive=False)
        with package.open("rb") as stream:
            response = connection.call(
                [
                    *PYTHON,
                    "-c",
                    remote_program(),
                    "stage",
                    context.run_key,
                    sha256_file(package),
                ],
                timeout=900,
                stdin=stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        if json.loads(response.stdout)["run_dir"] != remote_run:
            raise ValueError("Remote staging path mismatch")
    # Finalization needs a staged controller; arm it before start can submit a job.
    atomic_json(output / "client.json", client_state)
    remote_action(connection, remote_run, "start")
    wait_for_run(connection, remote_run, output, RUN_TIMEOUT)


def finalize_run(context, connection):
    output = context.output
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "client.json").exists():
        atomic_json(output / "finalize.json", {"status": "not-submitted"})
        return
    connection.deadline = time.monotonic() + FINALIZE_TIMEOUT
    info = preflight(connection, context)
    remote_run = str(PurePosixPath(info["root"]) / "runs" / context.run_key)
    saved = read_json(output / "client.json")
    for key, expected in {
        "run_key": context.run_key,
        "source_sha": context.sha,
        "remote_run": remote_run,
        "ssh_host": SSH_HOST,
    }.items():
        if saved.get(key) != expected:
            raise ValueError(f"Saved Actions run identity mismatch: {key}")
    try:
        state = remote_action(connection, remote_run, "finalize", 180)
        print(json.dumps(state), flush=True)
        atomic_json(output / "client-status.json", state)
    finally:
        collect(connection, remote_run, output, timeout=60)


def main():
    if sys.version_info < (3, 12):
        raise ValueError("The Slurm client requires Python 3.12 or newer")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "finalize"))
    args = parser.parse_args()
    context = ActionsContext.from_environment()
    connection = Connection(context)
    if args.action == "finalize":
        finalize_run(context, connection)
        return
    connection.deadline = time.monotonic() + RUN_TIMEOUT
    info = preflight(connection, context)
    if args.action == "preflight":
        print(json.dumps(info))
    else:
        start_run(context, connection, info)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Slurm CI: {error}", file=sys.stderr)
        if error.stderr:
            detail = error.stderr
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            print(detail[-8192:], file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"Slurm CI: {error}", file=sys.stderr)
        sys.exit(1)
