#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the frozen aggregate ladder and preserve exact collection evidence."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["aggregate"], required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    assert Path(args.python) == Path("/opt/dynamo/venv/bin/python3")
    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("ROCR_VISIBLE_DEVICES"), "Missing Slurm GPU mask"
    # Reject image-baked secondary masks rather than guessing their composition.
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    assert not os.environ.get("HIP_VISIBLE_DEVICES")
    os.chdir("/workspace")
    started_at = time.time()
    result = args.results
    (result / "test-results").mkdir(parents=True, exist_ok=True)
    os.environ.update(
        DYN_TEST_OUTPUT_PATH=str(result / "service-logs"),
        DYNAMO_CI_SERVICE_HOST="127.0.0.1",
        DYN_HTTP_HOST="127.0.0.1",
        DYN_SYSTEM_HOST="127.0.0.1",
    )
    contract = json.loads(Path(__file__).with_name("contract.json").read_text())
    request = json.loads((result / "request.json").read_text())
    model_content = json.loads(Path("/models/model-content.json").read_text())
    assert model_content["model"] == contract["model"]
    assert (
        Path("/models/hub/models--Qwen--Qwen3-0.6B/refs/main").read_text().strip()
        == contract["model"]["revision"]
    )
    # Full cache rehash protects explicit artifact reuse and offline resolution.
    sys.path.insert(0, str(Path(__file__).parent))
    spec = importlib.util.spec_from_file_location(
        "prepare_models", Path(__file__).with_name("prepare-models.py")
    )
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    files = models.inventory(Path("/models"))
    files.pop("model-content.json")
    assert files == model_content["files"]
    build = json.loads(Path("/opt/dynamo/ci-manifest.json").read_text())
    assert build["source_sha"] == request["source_sha"]
    validator = "/results/controller/barite_verify.py"
    subprocess.run(
        [
            args.python,
            validator,
            "provenance",
            "--run-dir",
            str(result),
            "--build-manifest",
            "/opt/dynamo/ci-manifest.json",
            "--output",
            str(result / "provenance.json"),
        ],
        check=True,
    )
    assert torch.version.hip and torch.version.cuda is None
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    architecture = torch.cuda.get_device_properties(0).gcnArchName
    assert architecture.split(":")[0] == "gfx942", "Expected allocated MI300X"
    x = torch.ones((2, 2), device="cuda")
    actual = (x @ x).cpu().tolist()
    torch.cuda.synchronize()
    assert actual == [[2.0, 2.0], [2.0, 2.0]]
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
