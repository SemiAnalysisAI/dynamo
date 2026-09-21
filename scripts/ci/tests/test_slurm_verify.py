# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure negative-path checks; no Slurm, network, or accelerator required."""

import ast
import copy
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slurm_verify as verify
from slurm_common import atomic_json, sha256_file


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.request = {
            "run_key": "test-run",
            "source_sha": "a" * 40,
            "archive_sha256": "b" * 64,
            "controller_sha": "c" * 40,
            "controller_bundle_sha256": "d" * 64,
        }
        self.image_build = {
            "platform": "linux/amd64",
            "runtime_dockerfile_sha256": "1" * 64,
            "test_dockerfile_sha256": "2" * 64,
            "runtime_image_id": "sha256:" + "3" * 64,
            "test_image_id": "sha256:" + "4" * 64,
            "runtime_sanity": {
                "status": "passed",
                "image_id": "sha256:" + "3" * 64,
                "checks": {"sanity_check": 0, "pip_check": 0},
            },
        }

    def archive(self, entries):
        files = []
        with tarfile.open(self.root / "source.tar", "w") as archive:
            for name, data, link in entries:
                item = tarfile.TarInfo(name)
                item.mode = 0o777 if link else 0o644
                item.size = 0 if link else len(data)
                if link:
                    item.type = tarfile.SYMTYPE
                    item.linkname = data.decode()
                archive.addfile(item, None if link else io.BytesIO(data))
                record = {
                    "path": name,
                    "type": "symlink" if link else "file",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                    "mode": item.mode,
                }
                if link:
                    record["linkname"] = item.linkname
                files.append(record)
        self.request["archive_sha256"] = sha256_file(self.root / "source.tar")
        atomic_json(self.root / "request.json", self.request)
        atomic_json(
            self.root / "source-manifest.json", {**self.request, "files": files}
        )

    def test_archive_roundtrip_and_hash(self):
        self.archive([("file", b"abc", False), ("alias", b"file", True)])
        verify.extract_source(self.root, self.root / "source")
        self.assertEqual((self.root / "source/alias").read_bytes(), b"abc")
        with (self.root / "source.tar").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify.extract_source(self.root, self.root / "other")

    def test_malicious_archives(self):
        for entries in (
            [("../escape", b"x", False)],
            [("link", b"../../escape", True)],
            [("link", b"target", True), ("link/child", b"x", False)],
        ):
            with self.subTest(entries=entries):
                self.archive(entries)
                with self.assertRaises(ValueError):
                    verify.extract_source(self.root, self.root / "source")
                self.assertFalse((self.root / "source").exists())

    def test_symlink_chain_cannot_escape_source(self):
        self.archive([("alias", b".", True), ("escape", b"alias/../outside", True)])
        with self.assertRaisesRegex(ValueError, "escaping symlink chain"):
            verify.extract_source(self.root, self.root / "source")
        self.assertFalse((self.root / "outside").exists())

    def test_source_manifest_mismatch(self):
        self.archive([("file", b"abc", False)])
        manifest = verify.read_json(self.root / "source-manifest.json")
        manifest["files"][0]["sha256"] = "0" * 64
        atomic_json(self.root / "source-manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest file mismatch"):
            verify.extract_source(self.root, self.root / "source")

    def test_stale_build_and_image_digest(self):
        build = {
            **self.request,
            "status": "passed",
            "wheels": ["wheel"],
            "image_build": self.image_build,
        }
        image = self.root / "image.sqsh"
        image.write_bytes(b"squashfs")
        manifest = verify.image_manifest(self.request, image, build)
        self.assertEqual(verify.verify_image(self.request, image, manifest), manifest)
        with self.assertRaisesRegex(ValueError, "source_sha mismatch"):
            verify.verify_image(
                {**self.request, "source_sha": "0" * 40}, image, manifest
            )
        image.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "image hash mismatch"):
            verify.verify_image(self.request, image, manifest)

    def test_build_requires_official_runtime_and_test_image_evidence(self):
        build = {
            **self.request,
            "status": "passed",
            "wheels": ["wheel"],
            "image_build": self.image_build,
        }
        verify.verify_build(build, self.request)
        invalid = []
        for key in self.image_build:
            evidence = copy.deepcopy(self.image_build)
            del evidence[key]
            invalid.append(evidence)
        for key, value in (
            ("platform", "linux/arm64"),
            ("runtime_dockerfile_sha256", "A" * 64),
            ("test_dockerfile_sha256", "not-a-digest"),
            ("runtime_image_id", "runtime:latest"),
            ("test_image_id", "4" * 64),
            ("test_image_id", self.image_build["runtime_image_id"]),
        ):
            invalid.append({**self.image_build, key: value})
        for key, value in (("status", "failed"), ("image_id", "sha256:" + "5" * 64)):
            evidence = copy.deepcopy(self.image_build)
            evidence["runtime_sanity"][key] = value
            invalid.append(evidence)
        for check in ("sanity_check", "pip_check"):
            for code in (None, 1, False, "0"):
                evidence = copy.deepcopy(self.image_build)
                evidence["runtime_sanity"]["checks"][check] = code
                invalid.append(evidence)
        evidence = copy.deepcopy(self.image_build)
        evidence["runtime_sanity"]["checks"]["dynamo_import"] = 1
        invalid.append(evidence)
        for evidence in (None, {}, *invalid):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                verify.verify_build({**build, "image_build": evidence}, self.request)

        trusted_path = Path(verify.__file__).parent / "rocm/contract.json"
        trusted = verify.read_json(trusted_path)
        spec = {
            **self.request,
            "contract_sha256": sha256_file(trusted_path),
            "base_uri": trusted["base_uri"],
            "model": trusted["model"],
        }
        with self.assertRaisesRegex(ValueError, "missing image build evidence"):
            verify.build_manifest(self.request, spec)

    def test_external_image_envelope_checks_both_identities(self):
        build = {
            **self.request,
            "status": "passed",
            "wheels": ["wheel"],
            "image_build": self.image_build,
        }
        envelope = {
            **self.request,
            "status": "passed",
            "sqsh_sha256": "e" * 64,
            "build": build,
        }
        self.assertIs(verify.build_from_image(self.request, envelope), build)
        for key in verify.identity(self.request):
            with self.subTest(key=key):
                wrong = copy.deepcopy(envelope)
                wrong[key] = "0" * len(wrong[key])
                with self.assertRaisesRegex(ValueError, key + " mismatch"):
                    verify.build_from_image(self.request, wrong)
                wrong = copy.deepcopy(envelope)
                wrong["build"][key] = "0" * len(wrong["build"][key])
                with self.assertRaisesRegex(ValueError, key + " mismatch"):
                    verify.build_from_image(self.request, wrong)
        for field, value in (
            ("status", "failed"),
            ("sqsh_sha256", "image:latest"),
            ("build", None),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.build_from_image(self.request, {**envelope, field: value})
        with self.assertRaisesRegex(ValueError, "reuse image mismatch"):
            verify.build_from_image(
                {**self.request, "reuse_image_sha": "f" * 64}, envelope
            )

        atomic_json(self.root / "request.json", self.request)
        atomic_json(self.root / "image-manifest.json", envelope)
        with (
            patch.object(
                sys,
                "argv",
                [
                    "slurm_verify.py",
                    "provenance",
                    "--run-dir",
                    str(self.root),
                    "--manifest",
                    str(self.root / "image-manifest.json"),
                ],
            ),
            patch.object(verify, "provenance") as provenance,
        ):
            verify.main()
            provenance.assert_called_once_with(self.request, build)
            atomic_json(
                self.root / "image-manifest.json",
                {**envelope, "controller_sha": "0" * 40},
            )
            provenance.reset_mock()
            with self.assertRaisesRegex(ValueError, "controller_sha mismatch"):
                verify.main()
            provenance.assert_not_called()

    def test_junit_empty_skipped_duplicate_failed_and_stale(self):
        path = self.root / "junit.xml"
        for content in (
            "<testsuite/>",
            '<testsuite><testcase classname="a" name="b"><skipped type="pytest.xfail"/></testcase></testsuite>',
            '<testsuite><testcase classname="a" name="b"/><testcase classname="a" name="b"/></testsuite>',
            '<testsuite><testcase classname="a" name="b"><failure/></testcase></testsuite>',
        ):
            path.write_text(content)
            with self.assertRaises(ValueError):
                verify.verify_junit(path, ["a::b"], 0)
        path.write_text('<testsuite><testcase classname="a" name="b"/></testsuite>')
        self.assertEqual(verify.verify_junit(path, ["a::b"], 0)["cases"], ["a::b"])
        os.utime(path, (1, 1))
        with self.assertRaisesRegex(ValueError, "stale JUnit"):
            verify.verify_junit(path, ["a::b"], 2)

    def test_wrong_import_origin_rejected(self):
        build = {
            **self.request,
            "status": "passed",
            "image_build": self.image_build,
            "wheels": [
                {"distribution": "candidate", "modules": ["candidate"], "record": {}}
            ],
        }
        root = self.root / "venv/site-packages"
        root.mkdir(parents=True)
        distribution = SimpleNamespace(locate_file=lambda _: root)
        module = SimpleNamespace(__file__=str(self.root / "workspace/candidate.py"))
        with (
            patch.object(verify.sys, "prefix", str(self.root / "venv")),
            patch.object(
                verify.importlib.metadata, "distribution", return_value=distribution
            ),
            patch.object(verify.importlib, "import_module", return_value=module),
            self.assertRaisesRegex(ValueError, "outside installed distribution"),
        ):
            verify.provenance(self.request, build)

    def test_imported_file_must_match_built_wheel_record(self):
        root = self.root / "venv/site-packages"
        root.mkdir(parents=True)
        module_path = root / "candidate.py"
        module_path.write_text("changed after wheel install")
        build = {
            **self.request,
            "status": "passed",
            "image_build": self.image_build,
            "wheels": [
                {
                    "distribution": "candidate",
                    "modules": ["candidate"],
                    "record": {"candidate.py": "0" * 64},
                }
            ],
        }
        distribution = SimpleNamespace(locate_file=lambda _: root)
        module = SimpleNamespace(__file__=str(module_path))
        with (
            patch.object(verify.sys, "prefix", str(self.root / "venv")),
            patch.object(
                verify.importlib.metadata, "distribution", return_value=distribution
            ),
            patch.object(verify.importlib, "import_module", return_value=module),
            self.assertRaisesRegex(ValueError, "module RECORD mismatch"),
        ):
            verify.provenance(self.request, build)

    def listener_evidence(self, suite="frontend"):
        ports = {
            name: 10000 + index
            for index, name in enumerate(
                ("nats", "etcd_client", "etcd_peer", "frontend", "system")
            )
        }
        evidence = {
            "status": "passed",
            "run_key": self.request["run_key"],
            "source_sha": self.request["source_sha"],
            "job_id": "123",
            "suite": suite,
            "expected_ports": ports,
            "listeners": [
                {
                    "role": role,
                    "port": port,
                    "address": "127.0.0.1",
                    "pid": 10,
                    "root_pid": 9,
                    "process_created_at": 1,
                    "captured_at": 2,
                }
                for role, port in ports.items()
            ],
        }

        evidence["service_listeners"] = [
            {
                **{key: value for key, value in entry.items() if key != "role"},
                "service_roles": ["frontend", "system"],
            }
            for entry in evidence["listeners"]
            if entry["role"] in ("frontend", "system")
        ]
        return evidence

    def test_listener_gate_rejects_missing_wildcard_stale_and_wrong_run(self):
        valid = self.listener_evidence()
        verify.verify_listener_evidence(valid, self.request, "frontend", "123", 0)
        for fault in ("missing", "wildcard", "stale", "identity", "extra"):
            evidence = self.listener_evidence()
            if fault == "missing":
                evidence["listeners"].pop()
            elif fault == "wildcard":
                evidence["listeners"][0]["address"] = "0.0.0.0"
            elif fault == "stale":
                evidence["listeners"][0]["captured_at"] = 0
            elif fault == "identity":
                evidence["job_id"] = "other"
            else:
                evidence["listeners"].append(
                    {**evidence["listeners"][0], "address": "::"}
                )
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                verify.verify_listener_evidence(
                    evidence, self.request, "frontend", "123", 1
                )

    def test_expanded_service_inventory_rejects_exposed_or_missing_listener(self):
        for fault in ("exposed", "empty", "owner", "endpoint"):
            evidence = self.listener_evidence()
            if fault == "exposed":
                evidence["service_listeners"].append(
                    {
                        **evidence["service_listeners"][0],
                        "port": 42207,
                        "address": "192.168.1.64",
                    }
                )
            elif fault == "empty":
                evidence["service_listeners"] = []
            elif fault == "owner":
                evidence["service_listeners"][0]["pid"] = 999
            else:
                evidence["service_listeners"][0]["port"] = 42207
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                verify.verify_listener_evidence(
                    evidence, self.request, "frontend", "123", 0
                )
        evidence = self.listener_evidence()
        evidence["service_listeners"].append(
            {**evidence["service_listeners"][0], "port": 42207, "address": "127.0.0.1"}
        )
        verify.verify_listener_evidence(evidence, self.request, "frontend", "123", 0)

    def test_service_capture_uses_only_anchor_pid_and_rejects_pid_reuse(self):
        source = Path(verify.__file__).parent / "rocm/pytest_driver.py"
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "service_listeners"
        )
        process = SimpleNamespace(
            create_time=lambda: 1,
            net_connections=lambda **_: [
                SimpleNamespace(
                    status="LISTEN",
                    laddr=SimpleNamespace(port=42207, ip="192.168.1.64"),
                )
            ],
        )
        visited = []

        def lookup(pid):
            visited.append(pid)
            return process

        namespace = {
            "psutil": SimpleNamespace(Process=lookup, CONN_LISTEN="LISTEN"),
            "time": SimpleNamespace(time=lambda: 2),
        }
        exec(  # noqa: S102 -- execute only the trusted capture helper
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        anchors = [
            {"pid": 11, "role": "frontend", "root_pid": 10, "process_created_at": 1}
        ]
        records = namespace["service_listeners"](anchors)
        self.assertEqual(visited, [11])
        self.assertEqual(records[0]["address"], "192.168.1.64")
        self.assertEqual(records[0]["service_roles"], ["frontend"])
        process.create_time = lambda: 3
        with self.assertRaisesRegex(ValueError, "PID was reused"):
            namespace["service_listeners"](anchors)

    def test_owned_listener_capture_filters_unrelated_ports(self):
        source = Path(verify.__file__).parent / "rocm/pytest_driver.py"
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "owned_listeners"
        )

        def connection(port, status="LISTEN"):
            return SimpleNamespace(
                status=status, laddr=SimpleNamespace(port=port, ip="127.0.0.1")
            )

        child = SimpleNamespace(
            pid=11,
            create_time=lambda: 1,
            net_connections=lambda **_: [
                connection(9999),
                connection(1234),
                connection(1234, "ESTABLISHED"),
            ],
        )
        root = SimpleNamespace(
            pid=10,
            create_time=lambda: 1,
            children=lambda **_: [child],
            net_connections=lambda **_: [],
        )
        namespace = {
            "psutil": SimpleNamespace(
                Process=lambda pid: root,
                CONN_LISTEN="LISTEN",
                NoSuchProcess=ProcessLookupError,
                ZombieProcess=ProcessLookupError,
            ),
            "time": SimpleNamespace(time=lambda: 2),
        }
        exec(  # noqa: S102 -- execute the trusted capture helper only
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        records = namespace["owned_listeners"](10, {"frontend": 1234})
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]["pid"], records[0]["port"]), (11, 1234))

    def test_workload_requires_all_frozen_cases_and_current_run(self):
        trusted_path = Path(verify.__file__).parent / "rocm/contract.json"
        trusted = verify.read_json(trusted_path)
        suites = {
            name: [
                node.split("::", 1)[0].removesuffix(".py").replace("/", ".")
                + "::"
                + node.split("::", 1)[1]
                for node in nodes
            ]
            for name, nodes in trusted["suites"].items()
        }
        runtime = {
            "run_key": self.request["run_key"],
            "source_sha": self.request["source_sha"],
            "started_at": 0,
            "suites": suites,
        }
        model_content = {"model": trusted["model"], "files": {}}
        model_hash = hashlib.sha256(
            (
                json.dumps(model_content, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest()
        runtime["model_manifest_sha256"] = model_hash
        atomic_json(
            self.root / "model-manifest.json",
            {**model_content, "model_manifest_sha256": model_hash},
        )
        atomic_json(self.root / "contract.json", runtime)
        atomic_json(
            self.root / "hip.json",
            {
                **self.request,
                "status": "passed",
                "hip": "7.2",
                "architecture": "gfx942",
                "device_count": 1,
                "rocr_visible_devices": "2",
                "result": [[2.0, 2.0], [2.0, 2.0]],
            },
        )
        build = {
            **self.request,
            "status": "passed",
            "image_build": self.image_build,
            "wheels": ["wheel"],
            "contract_sha256": sha256_file(trusted_path),
            "pytest_plugins": [["xdist", "xdist.plugin", "pytest-xdist", "3.8.0"]],
        }
        atomic_json(
            self.root / "provenance.json",
            {
                **self.request,
                "status": "passed",
                "created_at": 1,
                "build_manifest_sha256": verify.sha256_json(build),
            },
        )
        atomic_json(
            self.root / "image-manifest.json",
            {
                **self.request,
                "status": "passed",
                "sqsh_sha256": "e" * 64,
                "build": build,
            },
        )
        (self.root / "test-results").mkdir()
        for name, cases in suites.items():
            if name != "imports":
                atomic_json(
                    self.root / f"{name}-listeners.json", self.listener_evidence(name)
                )
            atomic_json(
                self.root / f"{name}-collection.json",
                {"nodeids": trusted["suites"][name], "plugins": ["xdist", "python"]},
            )
            body = "".join(
                f'<testcase classname="{case.split("::")[0]}" name="{case.split("::")[1]}"/>'
                for case in cases
            )
            (self.root / "test-results" / f"{name}.xml").write_text(
                "<testsuite>" + body + "</testsuite>"
            )
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
            result = verify.workload(self.request, self.root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                sum(len(suite["cases"]) for suite in result["junit"].values()), 8
            )
            hip = verify.read_json(self.root / "hip.json")
            for architecture in ("gfx9420", "gfx90a"):
                with self.subTest(architecture=architecture):
                    atomic_json(
                        self.root / "hip.json", {**hip, "architecture": architecture}
                    )
                    with self.assertRaisesRegex(ValueError, "GPU architecture"):
                        verify.workload(self.request, self.root)
            atomic_json(self.root / "hip.json", hip)
            atomic_json(self.root / "completed.json", result)
            # Controller-side verification requires no Slurm/GPU environment.
            with patch.dict(os.environ, clear=True):
                self.assertEqual(
                    verify.verify_collected(self.request, self.root), result
                )
                junit = self.root / "test-results/frontend.xml"
                original = junit.read_text()
                junit.write_text(original + "\n")
                with self.assertRaisesRegex(
                    ValueError, "collected workload mismatch: junit"
                ):
                    verify.verify_collected(self.request, self.root)
                junit.unlink()
                with self.assertRaises(FileNotFoundError):
                    verify.verify_collected(self.request, self.root)
                junit.write_text(original)
            runtime["suites"]["imports"] = []
            atomic_json(self.root / "contract.json", runtime)
            with self.assertRaisesRegex(ValueError, "trusted selection"):
                verify.workload(self.request, self.root)


class ServiceBindingTests(unittest.TestCase):
    def test_loopback_and_default_commands(self):
        # Load just these existing classes: importing conftest needs GPU test dependencies.
        source = Path(__file__).resolve().parents[3] / "tests/conftest.py"
        tree = ast.parse(source.read_text())
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in ("EtcdServer", "NatsServer")
        ]

        class Process:
            def __init__(self, **kwargs):
                self.command = kwargs["command"]

        with tempfile.TemporaryDirectory() as temporary:
            namespace = {
                "ManagedProcess": Process,
                "os": os,
                "tempfile": SimpleNamespace(mkdtemp=lambda **_: temporary),
                "allocate_ports": lambda *_: (12345, 12346),
                "allocate_port": lambda *_: 12347,
            }
            exec(  # noqa: S102 -- execute only the two trusted fixture classes
                compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"),
                namespace,
            )
            request = SimpleNamespace(node=SimpleNamespace(name="binding"))
            for port in (0, 2379):
                with patch.dict(os.environ, {"DYNAMO_CI_SERVICE_HOST": "127.0.0.1"}):
                    etcd = namespace["EtcdServer"](request, port=port)
                    nats = namespace["NatsServer"](request, port=0)
                    self.assertIn("--listen-peer-urls", etcd.command)
                    urls = [item for item in etcd.command if "http://" in item]
                    self.assertEqual(len(urls), 5)
                    self.assertTrue(all("http://127.0.0.1:" in item for item in urls))
                    self.assertEqual(
                        nats.command[nats.command.index("--addr") + 1], "127.0.0.1"
                    )
                    if port == 0:
                        self.assertIn("http://127.0.0.1:12345", urls)
                        self.assertIn("http://127.0.0.1:12346", urls)
            with patch.dict(os.environ, clear=True):
                etcd = namespace["EtcdServer"](request)
                nats = namespace["NatsServer"](request)
                self.assertIn("http://0.0.0.0:2379", etcd.command)
                self.assertNotIn("--listen-peer-urls", etcd.command)
                self.assertNotIn("--addr", nats.command)


if __name__ == "__main__":
    unittest.main()
