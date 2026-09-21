#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One isolated pytest process with installed-package provenance gates."""

import json
import os
import sys
import time
from importlib import import_module, metadata
from pathlib import Path

import psutil
import pytest

# The trusted controller is mounted outside Python site-packages.
sys.path.insert(0, "/results/controller")
from barite_common import atomic_json
from barite_verify import provenance, verify_listener_evidence


def owned_listeners(root_pid, expected_ports):
    """Inspect only this process tree and only explicitly requested service ports."""
    root = psutil.Process(root_pid)
    records = []
    for process in [root, *root.children(recursive=True)]:
        try:
            created = process.create_time()
            for connection in process.net_connections(kind="inet"):
                if connection.status != psutil.CONN_LISTEN:
                    continue
                for role, port in expected_ports.items():
                    if connection.laddr.port == port:
                        records.append(
                            {
                                "role": role,
                                "port": port,
                                "address": connection.laddr.ip,
                                "pid": process.pid,
                                "process_created_at": created,
                                "root_pid": root_pid,
                                "captured_at": time.time(),
                            }
                        )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # A short-lived worker child may exit while its stable service remains.
            continue
    return records


def main():
    suite, directory = sys.argv[1:]
    result = Path(directory)
    contract = json.loads(Path(__file__).with_name("contract.json").read_text())
    request = json.loads((result / "request.json").read_text())
    build = json.loads(Path("/opt/dynamo/ci-manifest.json").read_text())
    plugins = sorted(
        [e.name, e.value, e.dist.name, e.dist.version]
        for e in metadata.entry_points(group="pytest11")
    )
    assert plugins == build["pytest_plugins"], "Pytest plugin inventory changed"
    os.chdir("/workspace")
    # Required for repository tests.utils imports, not for Dynamo imports.
    sys.path.insert(0, "/workspace")

    class Evidence:
        def pytest_sessionstart(self, session):
            provenance(request, build)

        def pytest_collection_finish(self, session):
            actual = [item.nodeid for item in session.items]
            expected = contract["suites"][suite]
            if actual != expected:
                raise pytest.UsageError(
                    f"Frozen collection differs: {actual!r} != {expected!r}"
                )
            active = sorted(
                str(name)
                for name, plugin in session.config.pluginmanager.list_name_plugin()
                if plugin is not None
            )
            (result / (suite + "-collection.json")).write_text(
                json.dumps({"nodeids": actual, "plugins": active}, indent=2) + "\n"
            )
            provenance(request, build)

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_call(self, item):
            if suite == "imports":
                return (yield)
            nats, etcd = item.funcargs["runtime_services_dynamic_ports"]
            ports = item.funcargs["dynamo_dynamic_ports"]
            expected = {
                "nats": nats.port,
                "etcd_client": etcd.port,
                "etcd_peer": etcd.peer_port,
                "frontend": ports.frontend_port,
                "system": ports.system_ports[0],
            }
            evidence = {
                "run_key": request["run_key"],
                "source_sha": request["source_sha"],
                "job_id": os.environ["SLURM_JOB_ID"],
                "suite": suite,
                "expected_ports": expected,
                "listeners": [],
                "status": "collecting",
            }
            evidence["listeners"] += owned_listeners(nats.proc.pid, {"nats": nats.port})
            evidence["listeners"] += owned_listeners(
                etcd.proc.pid, {"etcd_client": etcd.port, "etcd_peer": etcd.peer_port}
            )
            path = result / (suite + "-listeners.json")
            atomic_json(path, evidence)

            def capture_services(root_pid):
                services = owned_listeners(
                    root_pid,
                    {"frontend": ports.frontend_port, "system": ports.system_ports[0]},
                )
                evidence["listeners"] = [
                    entry
                    for entry in evidence["listeners"]
                    if entry["role"] not in ("frontend", "system")
                ] + services
                evidence["status"] = "passed"
                atomic_json(path, evidence)
                verify_listener_evidence(
                    evidence, request, suite, evidence["job_id"], 0
                )

            if suite == "frontend":
                capture_services(os.getpid())
                return (yield)
            engine = import_module("tests.utils.engine_process").EngineProcess
            original = engine.check_response

            def checked_response(process, payload, response):
                outcome = original(process, payload, response)
                capture_services(process.proc.pid)
                return outcome

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(engine, "check_response", checked_response)
                outcome = yield
            verify_listener_evidence(evidence, request, suite, evidence["job_id"], 0)
            return outcome

    options = [
        "--rootdir=/workspace",
        "-c",
        "/workspace/pyproject.toml",
        "-n",
        "0",
        "-xvv",
        "--import-mode=importlib",
        "--junitxml=" + str(result / "test-results" / (suite + ".xml")),
    ]
    if suite != "imports":
        options += ["--models-dir=/models"]
    if suite == "frontend":
        options += ["--timeout=600"]
    raise SystemExit(
        int(pytest.main(options + contract["suites"][suite], plugins=[Evidence()]))
    )


if __name__ == "__main__":
    main()
