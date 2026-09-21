#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the frozen aggregate ladder and preserve exact collection evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Isolated Python omits the script directory; add only the trusted controller.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm.prepare_models import inventory
from slurm_verify import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["aggregate"], required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    require(
        Path(args.python) == Path("/opt/dynamo/venv/bin/python3"),
        "Tests require the image's overlay interpreter",
    )
    require(os.environ.get("SLURM_JOB_ID"), "Tests require a Slurm allocation")
    require(os.environ.get("ROCR_VISIBLE_DEVICES"), "Missing Slurm GPU mask")
    # Reject image-baked secondary masks rather than guessing their composition.
    require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "Unexpected CUDA GPU mask")
    require(not os.environ.get("HIP_VISIBLE_DEVICES"), "Unexpected HIP GPU mask")
    os.chdir("/workspace")
    started_at = time.time()
    result = args.results
    (result / "test-results").mkdir(parents=True, exist_ok=True)
    os.environ.update(
        DYN_TEST_OUTPUT_PATH=str(result / "service-logs"),
        DYNAMO_CI_SERVICE_HOST="127.0.0.1",
        DYN_HTTP_HOST="127.0.0.1",
        DYN_SYSTEM_HOST="127.0.0.1",
        # All cooperating Dynamo services share this one-node allocation.
        DYN_TCP_RPC_HOST="127.0.0.1",
        DYN_TCP_RESPONSE_STREAM_HOST="127.0.0.1",
        # Direct ZMQ publishers bind a wildcard even with a loopback advertised
        # host. Use the existing loopback NATS service for this one-node lane.
        DYN_EVENT_PLANE="nats",
    )
    contract = json.loads(Path(__file__).with_name("contract.json").read_text())
    request = json.loads((result / "request.json").read_text())
    model_content = json.loads(Path("/models/model-content.json").read_text())
    require(model_content["model"] == contract["model"], "Model contract mismatch")
    model_cache = Path("/models/hub") / (
        "models--" + contract["model"]["repo_id"].replace("/", "--")
    )
    require(
        (model_cache / "refs/main").read_text().strip()
        == contract["model"]["revision"],
        "Cached model revision mismatch",
    )
    # Full cache rehash protects explicit artifact reuse and offline resolution.
    files = inventory(Path("/models"))
    files.pop("model-content.json")
    require(files == model_content["files"], "Cached model content mismatch")
    validator = "/results/controller/slurm_verify.py"
    subprocess.run(
        [
            args.python,
            validator,
            "provenance",
            "--run-dir",
            str(result),
            "--manifest",
            str(result / "image-manifest.json"),
            "--output",
            str(result / "provenance.json"),
        ],
        check=True,
    )
    require(torch.version.hip and torch.version.cuda is None, "ROCm PyTorch required")
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "Exactly one allocated HIP device is required",
    )
    architecture = torch.cuda.get_device_properties(0).gcnArchName
    require(architecture.split(":")[0] == "gfx942", "Expected allocated MI300X")
    x = torch.ones((2, 2), device="cuda")
    actual = (x @ x).cpu().tolist()
    torch.cuda.synchronize()
    require(actual == [[2.0, 2.0], [2.0, 2.0]], "HIP arithmetic failed")
    hip = {
        "status": "passed",
        "run_key": request["run_key"],
        "source_sha": request["source_sha"],
        "architecture": architecture,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0),
        "rocr_visible_devices": os.environ["ROCR_VISIBLE_DEVICES"],
        "result": actual,
    }
    (result / "hip.json").write_text(json.dumps(hip, indent=2) + "\n")

    def junit_id(node):
        path, name = node.split("::", 1)
        return path.removesuffix(".py").replace("/", ".") + "::" + name

    runtime_contract = {
        "run_key": request["run_key"],
        "source_sha": request["source_sha"],
        "started_at": started_at,
        "model_manifest_sha256": hashlib.sha256(
            Path("/models/model-content.json").read_bytes()
        ).hexdigest(),
        "suites": {
            name: [junit_id(node) for node in nodes]
            for name, nodes in contract["suites"].items()
        },
    }
    (result / "contract.json").write_text(json.dumps(runtime_contract, indent=2) + "\n")
    subprocess.run([args.python, "-m", "pip", "check"], check=True)
    for suite in contract["suites"]:
        subprocess.run(
            [
                args.python,
                "-I",
                str(Path(__file__).with_name("pytest_driver.py")),
                suite,
                str(result),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
