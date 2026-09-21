#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record installed artifacts from the built test image without changing it."""

import argparse
import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm_common import atomic_json, read_json, sha256_file
from slurm_verify import build_manifest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results
    request = read_json(results / "request.json")
    contract_path = Path(__file__).with_name("contract.json")
    contract = read_json(contract_path)
    require(
        os.environ.get("DYNAMO_COMMIT_SHA") == request["source_sha"],
        "Container source revision differs from the request",
    )
    require(
        Path(sys.prefix).resolve() == Path("/opt/dynamo/venv"),
        "Expected the built image's Dynamo interpreter",
    )
    import torch
    import vllm

    import dynamo._core

    require(torch.version.hip and not torch.version.cuda, "ROCm PyTorch required")
    require(
        vllm.__version__ == contract["tools"]["vllm"],
        "Container vLLM version differs from the contract",
    )
    for name, command in {
        "installed-packages.log": [sys.executable, "-m", "pip", "freeze", "--all"],
        "native-packages.log": ["dpkg-query", "-W"],
        "native-linkage.log": ["ldd", dynamo._core.__file__],
        "pip-check.log": [sys.executable, "-m", "pip", "check"],
    }.items():
        completed = subprocess.run(command, text=True, capture_output=True)
        (results / name).write_text(completed.stdout + completed.stderr)
        require(completed.returncode == 0, f"Image inspection failed: {name}")
    linkage = (results / "native-linkage.log").read_text()
    require("not found" not in linkage, "Unresolved native library dependency")
    require(
        not re.search(r"lib(?:cuda|cudart|nixl)\S*\s+=>", linkage),
        "Unexpected CUDA/native NIXL linkage",
    )
    toolchain = read_json(Path("/opt/dynamo/build-toolchain.json"))
    for name, prefix, version in (
        ("rustc", "rustc ", contract["tools"]["rust"]),
        ("maturin", "maturin ", contract["tools"]["maturin"]),
        ("protoc", "libprotoc ", contract["tools"]["protoc"]),
    ):
        require(
            toolchain[name].splitlines()[0].split(" ")[:2]
            == (prefix + version).split(),
            f"Built image {name} differs from the qualification contract",
        )
    spec = {
        "source_sha": request["source_sha"],
        "archive_sha256": request["archive_sha256"],
        "base_uri": contract["base_uri"],
        "contract_sha256": sha256_file(contract_path),
        "model": contract["model"],
        "pytest_plugins": sorted(
            [entry.name, entry.value, entry.dist.name, entry.dist.version]
            for entry in importlib.metadata.entry_points(group="pytest11")
        ),
        "toolchain": toolchain,
        "features": ["bindings-default", "linux-llm-default"],
        "native": {"mode": "upstream-dlopen-fallback", "linkage": linkage},
        "image_build": read_json(results / "image-build.json"),
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
        wheels = list(Path("/opt/dynamo/dist").glob(pattern))
        require(len(wheels) == 1, f"Expected exactly one built wheel: {pattern}")
        spec["wheels"].append(
            {"path": str(wheels[0]), "distribution": distribution, "modules": modules}
        )
    atomic_json(results / "build-spec.json", spec)
    atomic_json(results / "build-manifest.json", build_manifest(request, spec))


if __name__ == "__main__":
    main()
