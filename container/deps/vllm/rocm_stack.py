# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze the inherited accelerator packages and check their installed origins."""

import argparse
import hashlib
import json
import re
from importlib import import_module, metadata
from pathlib import Path

PREFIX = Path("/opt/dynamo")


def stack():
    result = {}
    for name in ("torch", "vllm"):
        module = import_module(name)
        path = Path(module.__file__).resolve()
        result[name] = {
            "version": metadata.version(name),
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if name == "torch" and (
            not module.version.hip or module.version.cuda is not None
        ):
            raise ValueError("The base image must provide ROCm PyTorch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "verify"))
    args = parser.parse_args()
    manifest = PREFIX / "rocm-base.json"
    actual = stack()
    if args.action == "verify":
        if actual != json.loads(manifest.read_text()):
            raise ValueError("The inherited Torch/vLLM stack was replaced or modified")
        return
    protected = ("torch", "vllm", "triton", "pytorch-triton", "amd", "rocm")
    constraints = sorted(
        f"{distribution.name}=={distribution.version}"
        for distribution in metadata.distributions()
        if re.sub(r"[-_.]+", "-", distribution.name.lower()).startswith(protected)
    )
    PREFIX.mkdir(parents=True, exist_ok=True)
    (PREFIX / "rocm-constraints.txt").write_text("\n".join(constraints) + "\n")
    manifest.write_text(json.dumps(actual, indent=2) + "\n")


if __name__ == "__main__":
    main()
