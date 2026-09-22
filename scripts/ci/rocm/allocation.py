#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the allocated node's Slurm configuration and require device isolation."""

import argparse
import hashlib
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# The allocation executes this verified controller, not candidate source.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm_common import atomic_json, verify_controller


def read_evidence(path):
    try:
        data = path.read_bytes()
        return {
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "text": data.decode("utf-8"),
        }
    except (OSError, UnicodeError) as error:
        return {"path": str(path), "error": str(error)}


def effective_config(name):
    attempts = []
    for directory in ("/run/slurm/conf", "/var/spool/slurmd/conf-cache"):
        evidence = read_evidence(Path(directory) / name)
        if "text" in evidence:
            evidence["unavailable_alternatives"] = attempts
            return evidence
        attempts.append(evidence)
    return {"error": "No readable effective configuration", "attempts": attempts}


def capture(command, timeout=15):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def assignments(text):
    # Last assignment wins, as in the Slurm configuration parser.
    return dict(
        re.findall(
            r"([A-Za-z][A-Za-z0-9]*)\s*=\s*([^\s]+)",
            "\n".join(line.split("#", 1)[0] for line in text.splitlines()),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    verify_controller(run)
    atomic_json(
        run / "allocation.json",
        {
            k: os.environ.get(k)
            for k in (
                "SLURM_JOB_ID",
                "SLURMD_NODENAME",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_GPUS",
                "SLURM_RESTART_COUNT",
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    )

    cgroup = effective_config("cgroup.conf")
    gres = effective_config("gres.conf")
    config = capture(["scontrol", "show", "config"])
    # Retain the relevant scheduler settings, not an unrestricted configuration dump.
    settings = {
        key: value
        for key, value in assignments(config.get("stdout", "")).items()
        if key in ("TaskPlugin", "ProctrackType", "GresTypes", "SLURM_CONF")
    }
    config["settings"] = settings
    config.pop("stdout", None)
    node = os.environ["SLURMD_NODENAME"]
    gres_positive = False
    gres_selectors_valid = True
    gres_selector_evidence = []
    for line in gres.get("text", "").splitlines():
        values = assignments(line)
        selector = values.get("NodeName")
        if selector is not None and selector != node and selector != "ALL":
            expanded = capture(["scontrol", "show", "hostnames", selector])
            gres_selector_evidence.append(expanded)
            hosts = expanded.get("stdout", "").splitlines()
            if (
                expanded.get("returncode") != 0
                or not hosts
                or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", host) for host in hosts)
            ):
                gres_selectors_valid = False
                continue
            if node not in hosts:
                continue
        if values.get("AutoDetect", "").lower() == "rsmi":
            gres_positive = True
        elif "AutoDetect" in values:
            gres_positive = False
        if values.get("Name") == "gpu" and (
            values.get("File") or values.get("MultipleFiles")
        ):
            gres_positive = True
    checks = {
        "constrain_devices": assignments(cgroup.get("text", ""))
        .get("ConstrainDevices", "")
        .lower()
        == "yes",
        "task_cgroup": config.get("returncode") == 0
        and "task/cgroup" in settings.get("TaskPlugin", "").split(","),
        "gres_device_mapping_configured": gres_positive and gres_selectors_valid,
    }
    environment = {
        "schema_version": 1,
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": node,
        "uname": platform.uname()._asdict(),
        "amdgpu_version": read_evidence(Path("/sys/module/amdgpu/version")),
        "cgroup_membership": read_evidence(Path("/proc/self/cgroup")),
        "cgroup_config": cgroup,
        "gres_config": gres,
        "scheduler_config": config,
        "gres_selector_expansion": gres_selector_evidence,
        "scheduler_node": capture(["scontrol", "show", "node", "-o", node]),
        "enroot_version": capture(["enroot", "version"], timeout=10),
        "device_isolation_checks": checks,
        "evidence_scope": "Configuration evidence only; RSMI resolves device mappings automatically. "
        "This is not an empirical test of denied access to unallocated devices.",
    }
    atomic_json(run / "environment.json", environment)
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(
            "Required Slurm device-isolation configuration missing: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
