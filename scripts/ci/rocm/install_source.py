#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build image-local source wheels; all heavy work runs in an allocated rootfs."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import tomllib

ROOT = Path("/workspace")
PREFIX = Path("/opt/dynamo")
PYTHON = PREFIX / "venv/bin/python3"


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def fetch(url, path):
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def archive(url, destination, checksum_url, expected_sha256, member=None):
    path = PREFIX / url.rsplit("/", 1)[-1]
    fetch(url, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Pinned archive digest mismatch: {url}")
    if checksum_url:
        checksum = path.with_suffix(".checksums")
        fetch(checksum_url, checksum)
        lines = checksum.read_text().splitlines()
        expected = next(line.split()[0] for line in lines if path.name in line)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {url}")
    with tarfile.open(path) as bundle:
        bundle.extractall(destination, filter="data")
    path.unlink()
    return destination / (
        member or path.name.removesuffix(".tar.gz").removesuffix(".tar.xz")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--base-uri", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch("[0-9a-f]{40}", args.source_sha)
    assert re.fullmatch("[0-9a-f]{64}", args.archive_sha256)
    assert os.environ.get("SLURM_JOB_ID"), "Build must run inside a Slurm allocation"
    contract_path = Path(__file__).with_name("contract.json")
    contract = json.loads(contract_path.read_text())
    assert args.base_uri == contract["base_uri"]
    PREFIX.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    # Compiler outputs and caches belong in the allocated node-local source copy.
    cache = ROOT / ".ci-build-cache"
    cache.mkdir(exist_ok=True)
    os.environ["CARGO_HOME"] = str(cache / "cargo")
    os.environ["PIP_CACHE_DIR"] = str(cache / "pip")
    os.environ["TMPDIR"] = str(cache / "tmp")
    Path(os.environ["TMPDIR"]).mkdir(exist_ok=True)
    # Bindgen, native TLS/protobuf and C++ wrapper compilation are unconditional.
    run("apt-get", "update")
    run(
        "apt-get",
        "install",
        "-y",
        "--no-install-recommends",
        "build-essential",
        "clang",
        "libclang-dev",
        "pkg-config",
        "libssl-dev",
        "cmake",
        "python3-dev",
        "python3-venv",
        "curl",
        "ca-certificates",
        "xz-utils",
        "git",
        "libnuma-dev",
        "libudev-dev",
        "libcairo2-dev",
    )
    run("python3", "-m", "venv", "--system-site-packages", PREFIX / "venv")
    os.environ["PATH"] = (
        str(PREFIX / "venv/bin")
        + ":"
        + str(PREFIX / "rust/bin")
        + ":"
        + os.environ["PATH"]
    )
    os.environ["VIRTUAL_ENV"] = str(PREFIX / "venv")
    # Constrain engine, Torch, Triton and ROCm packages before resolving tools.
    before_code = "import json,torch,vllm; print(json.dumps({m.__name__:{'version':m.__version__,'file':m.__file__} for m in (torch,vllm)}))"
    before = subprocess.check_output([str(PYTHON), "-I", "-c", before_code], text=True)
    assert json.loads(before)["vllm"]["version"] == contract["tools"]["vllm"]
    constraints = PREFIX / "base-constraints.txt"
    base_freeze = subprocess.check_output(
        [str(PYTHON), "-m", "pip", "list", "--format=freeze"], text=True
    )
    protected = ("torch", "vllm", "triton", "pytorch-triton", "amd", "rocm")
    constraints.write_text(
        "\n".join(
            line
            for line in base_freeze.splitlines()
            if line.lower().startswith(protected)
        )
        + "\n"
    )
    pip = [PYTHON, "-m", "pip", "install", "--constraint", constraints]
    run(
        *pip,
        "maturin==" + contract["tools"]["maturin"],
        "hatchling==" + contract["tools"]["hatchling"],
        "build==1.3.0",
        "patchelf==0.17.2.4",
        "pyyaml==6.0.3",
    )
    rust = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"][
        "channel"
    ]
    assert rust == contract["tools"]["rust"]
    url = (
        f"https://static.rust-lang.org/dist/rust-{rust}-x86_64-unknown-linux-gnu.tar.xz"
    )
    extracted = archive(url, PREFIX, url + ".sha256", contract["tool_sha256"]["rust"])
    run(
        "bash",
        extracted / "install.sh",
        "--prefix=" + str(PREFIX / "rust"),
        "--without=rust-docs",
    )
    shutil.rmtree(extracted)
    # Ubuntu 22.04 protoc 3.12 cannot compile relay.proto's proto3 optional
    # fields. Use a pinned compiler plus its matching well-known-type includes.
    protoc_version = contract["tools"]["protoc"]
    protoc_zip = PREFIX / f"protoc-{protoc_version}-linux-x86_64.zip"
    fetch(
        f"https://github.com/protocolbuffers/protobuf/releases/download/v{protoc_version}/{protoc_zip.name}",
        protoc_zip,
    )
    if (
        hashlib.sha256(protoc_zip.read_bytes()).hexdigest()
        != contract["tool_sha256"]["protoc"]
    ):
        raise RuntimeError("Pinned protoc archive digest mismatch")
    protoc_root = PREFIX / "protoc"
    with zipfile.ZipFile(protoc_zip) as bundle:
        for member in bundle.namelist():
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise RuntimeError("Invalid protoc archive path")
        bundle.extractall(protoc_root)
    protoc_zip.unlink()
    protoc = protoc_root / "bin/protoc"
    protoc.chmod(0o755)
    os.environ["PROTOC"] = str(protoc)
    os.environ["PROTOC_INCLUDE"] = str(protoc_root / "include")
    os.environ["PATH"] = str(protoc_root / "bin") + ":" + os.environ["PATH"]
    actual_protoc = subprocess.check_output(
        [str(protoc), "--version"], text=True
    ).strip()
    if actual_protoc != f"libprotoc {protoc_version}":
        raise RuntimeError(f"Unexpected protoc version: {actual_protoc}")
    (args.manifest.parent / "protoc-version.log").write_text(actual_protoc + "\n")
    # Official nixl-sys 1.3.2 fallback builds a dlopen wrapper, not a Python
    # monkeypatch. Missing native NIXL is explicit: transport/KVBM remain unqualified.
    # cudarc 0.19.8 defaults to dynamic loading and accepts this ABI selector;
    # selecting declarations does not install CUDA or emulate a CUDA device.
    os.environ["NIXL_PREFIX"] = str(PREFIX / "no-native-nixl")
    os.environ["CUDARC_CUDA_VERSION"] = "13030"
    os.environ.pop("NIXL_NO_STUBS_FALLBACK", None)
    os.environ["CARGO_TARGET_DIR"] = "/workspace/.ci-cargo-target"
    dist = PREFIX / "dist"
    dist.mkdir(exist_ok=True)
    run(
        "cargo",
        "tree",
        "--locked",
        "--manifest-path",
        ROOT / "lib/bindings/python/Cargo.toml",
        "-e",
        "features",
        stdout=(args.manifest.parent / "cargo-features.log").open("w"),
    )
    run(
        PYTHON,
        "-m",
        "maturin",
        "build",
        "--locked",
        "--release",
        "--auditwheel",
        "skip",
        "--manifest-path",
        ROOT / "lib/bindings/python/Cargo.toml",
        "--out",
        dist,
    )
    run(PYTHON, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist, ROOT)
    # No backend extras: those request CUDA NIXL/CuPy wheels.
    run(
        PYTHON,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--no-deps",
        "--no-index",
        "--find-links",
        "https://pypi.nvidia.com/aisimulate/",
        "--dest",
        dist,
        "-r",
        ROOT / "container/deps/requirements.aisimulate.txt",
    )
    # Resolve all requirements together. In particular, make inherited vLLM an
    # explicit root so pip honors its grpcio pin when resolving AISimulate.
    run(
        *pip,
        *sorted(dist.glob("*.whl")),
        "vllm",
        "-r",
        ROOT / "container/deps/requirements.test.txt",
        "-r",
        Path(__file__).with_name("requirements.compat.txt"),
    )
    after = subprocess.check_output([str(PYTHON), "-I", "-c", before_code], text=True)
    assert json.loads(before) == json.loads(after), "Base engine version/origin changed"
    context = (ROOT / "container/context.yaml").read_text()
    for name in ("nats", "etcd"):
        version = contract["tools"][name]
        assert re.search(
            r"^  " + name + r"_version: " + re.escape(version) + r"$",
            context,
            re.MULTILINE,
        )
        if name == "nats":
            filename = f"nats-server-{version}-linux-amd64.tar.gz"
            url = f"https://github.com/nats-io/nats-server/releases/download/{version}/{filename}"
            unpacked = archive(
                url,
                PREFIX,
                f"https://github.com/nats-io/nats-server/releases/download/{version}/SHA256SUMS",
                contract["tool_sha256"]["nats"],
            )
            run(
                "install",
                "-m",
                "755",
                unpacked / "nats-server",
                PREFIX / "venv/bin/nats-server",
            )
        else:
            filename = f"etcd-{version}-linux-amd64.tar.gz"
            url = f"https://github.com/etcd-io/etcd/releases/download/{version}/{filename}"
            unpacked = archive(
                url,
                PREFIX,
                f"https://github.com/etcd-io/etcd/releases/download/{version}/SHA256SUMS",
                contract["tool_sha256"]["etcd"],
            )
            for executable in ("etcd", "etcdctl"):
                run(
                    "install",
                    "-m",
                    "755",
                    unpacked / executable,
                    PREFIX / "venv/bin" / executable,
                )
    (args.manifest.parent / "installed-packages.log").write_text(
        subprocess.check_output(
            [str(PYTHON), "-m", "pip", "freeze", "--all"], text=True
        )
    )
    (args.manifest.parent / "native-packages.log").write_text(
        subprocess.check_output(["dpkg-query", "-W"], text=True)
    )
    core = subprocess.check_output(
        [str(PYTHON), "-I", "-c", "import dynamo._core; print(dynamo._core.__file__)"],
        text=True,
    ).strip()
    linkage = subprocess.check_output(["ldd", core], text=True)
    assert "not found" not in linkage
    assert not re.search(r"lib(?:cuda|cudart|nixl)\S*\s+=>", linkage), (
        "Unexpected CUDA/native NIXL linkage"
    )
    (args.manifest.parent / "native-linkage.log").write_text(linkage)
    run(PYTHON, "-m", "pip", "check")
    plugin_code = "import importlib.metadata as m,json; print(json.dumps(sorted((e.name,e.value,e.dist.name,e.dist.version) for e in m.entry_points(group='pytest11'))))"
    plugins = json.loads(
        subprocess.check_output([str(PYTHON), "-I", "-c", plugin_code], text=True)
    )
    spec = {
        "source_sha": args.source_sha,
        "archive_sha256": args.archive_sha256,
        "base_uri": args.base_uri,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "model": contract["model"],
        "pytest_plugins": plugins,
        "toolchain": contract["tools"],
        "features": ["bindings-default", "linux-llm-default"],
        "native": {
            "nixl_sys": "1.3.2",
            "mode": "upstream-dlopen-fallback",
            "cudarc": "0.19.8",
            "cuda_declarations": "13.3",
            "linkage": linkage,
        },
        "system_modules": ["torch", "vllm"],
        "wheels": [],
    }
    for pattern, distribution, modules in [
        (
            "ai_dynamo_runtime-*.whl",
            "ai-dynamo-runtime",
            ["dynamo._core", "dynamo.runtime", "dynamo.llm"],
        ),
        ("ai_dynamo-*.whl", "ai-dynamo", ["dynamo.frontend", "dynamo.vllm"]),
    ]:
        (wheel,) = dist.glob(pattern)
        spec["wheels"].append(
            {"path": str(wheel), "distribution": distribution, "modules": modules}
        )
    spec_path = args.manifest.parent / "build-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    run(
        PYTHON,
        "/results/controller/barite_verify.py",
        "build-manifest",
        "--run-dir",
        args.manifest.parent,
        "--build-spec",
        spec_path,
        "--output",
        args.manifest,
    )
    run("cp", args.manifest, PREFIX / "ci-manifest.json")


if __name__ == "__main__":
    main()
