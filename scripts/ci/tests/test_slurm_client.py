# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline Actions identity, staging, finalization and false-success regressions."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slurm_client as client
import slurm_remote as remote


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkout(self, files):
        source = self.root / "source"
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        for name, content in files.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Offline Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        return source, client.git(source, "rev-parse", "HEAD").decode().strip()

    def actions_environment(self, workspace=None, sha=None):
        workspace = workspace or self.root / "workspace"
        workspace.mkdir(exist_ok=True)
        runner_temp = self.root / "runner-temp"
        runner_temp.mkdir(exist_ok=True)
        return {
            "GITHUB_ACTIONS": "true",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF_TYPE": "branch",
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_SHA": sha or "a" * 40,
            "GITHUB_WORKFLOW_SHA": sha or "a" * 40,
            "GITHUB_RUN_ID": "12345",
            "GITHUB_RUN_ATTEMPT": "2",
            "SLURM_UID": "20011",
            "PREFERRED_PARTITION": "compute-1",
            "FALLBACK_PARTITION": "compute-0",
        }

    def actions_context(self, environment=None):
        environment = environment or self.actions_environment()
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                client, "REPO", Path(environment["GITHUB_WORKSPACE"]).resolve()
            ),
        ):
            return client.ActionsContext.from_environment()

    def evidence(self):
        request = {
            "run_key": "gh-12345-2",
            "source_sha": "a" * 40,
            "archive_sha256": "b" * 64,
            "controller_sha": "c" * 40,
            "controller_bundle_sha256": "d" * 64,
        }
        values = {
            "request": request,
            "expected-request": request.copy(),
            "image-manifest": {"sqsh_sha256": "e" * 64},
            "controller-result": {"status": "terminal"},
            "receipt": {
                "job_id": "123",
                "run_key": "gh-12345-2",
                "source_sha": "a" * 40,
            },
            "terminal": {
                "job_id": "123",
                "run_key": "gh-12345-2",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "restarts": 0,
            },
            "wait-result": {"job_id": "123", "returncode": 0, "missing_record": False},
            "completed": {
                **request,
                "job_id": "123",
                "status": "passed",
                "sqsh_sha256": "e" * 64,
                "model_manifest_sha256": "f" * 64,
                "contract_sha256": "1" * 64,
            },
        }
        for name, data in values.items():
            client.atomic_json(self.root / (name + ".json"), data)
        return values

    def test_complete_matching_evidence_passes(self):
        self.evidence()
        with patch.object(client, "verify_collected") as verify:
            self.assertEqual(client.verdict(self.root)["status"], "passed")
            verify.assert_called_once()

    def test_collected_request_cannot_replace_actions_intent(self):
        values = self.evidence()
        values["request"]["source_sha"] = "0" * 40
        client.atomic_json(self.root / "request.json", values["request"])
        with self.assertRaisesRegex(ValueError, "original Actions request"):
            client.verdict(self.root)

    def test_missing_image_or_original_request_fails(self):
        for filename in ("image-manifest.json", "expected-request.json"):
            self.evidence()
            (self.root / filename).unlink()
            with self.subTest(filename=filename), self.assertRaises(FileNotFoundError):
                client.verdict(self.root)

    def test_archive_deterministic_across_mtime_changes(self):
        source, sha = self.checkout({"z.txt": b"last", "a.txt": b"first"})
        first = client.source_archive(source, self.root / "first.tar", sha)
        os.utime(source / "a.txt", (123456, 123456))
        second = client.source_archive(source, self.root / "second.tar", sha)
        self.assertEqual(first, second)
        self.assertEqual(
            (self.root / "first.tar").read_bytes(),
            (self.root / "second.tar").read_bytes(),
        )

    def test_lfs_pointer_never_staged_as_source(self):
        source, sha = self.checkout(
            {
                "weights": b"version https://git-lfs.github.com/spec/v1\noid sha256:"
                + b"a" * 64
                + b"\nsize 99\n"
            }
        )
        with self.assertRaisesRegex(ValueError, "Unmaterialized LFS"):
            client.source_archive(source, self.root / "source.tar", sha)

    def test_dirty_checkout_rejected(self):
        source, sha = self.checkout({"file": b"clean"})
        (source / "file").write_bytes(b"dirty")
        with self.assertRaisesRegex(ValueError, "clean"):
            client.source_archive(source, self.root / "source.tar", sha)

    def test_actions_identity_and_storage_are_derived_from_runner(self):
        context = self.actions_context()
        self.assertEqual(context.sha, "a" * 40)
        self.assertEqual(context.run_key, "gh-12345-2")
        self.assertEqual(context.expected_uid, 20011)
        self.assertEqual(context.preferred_partition, "compute-1")
        self.assertEqual(context.fallback_partition, "compute-0")
        self.assertEqual(context.output, context.runner_temp / "dynamo-rocm/evidence")
        connection = client.Connection(context)
        self.assertEqual(
            connection.prefix[-3:],
            [
                "-F",
                str(context.runner_temp / "dynamo-rocm/ssh/config"),
                "ci-slurm-login",
            ],
        )

    def test_only_workflow_dispatch_context_is_accepted(self):
        environment = self.actions_environment()
        for field, value in (
            ("GITHUB_ACTIONS", "false"),
            ("GITHUB_ACTIONS", ""),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_EVENT_NAME", "pull_request_target"),
            ("GITHUB_REF_TYPE", "tag"),
            ("GITHUB_RUN_ID", "123;other"),
            ("GITHUB_RUN_ATTEMPT", "0"),
            ("GITHUB_SHA", "main"),
            ("GITHUB_WORKFLOW_SHA", "b" * 40),
            ("SLURM_UID", "0"),
            ("PREFERRED_PARTITION", "auto"),
            ("FALLBACK_PARTITION", "compute-1"),
            ("FALLBACK_PARTITION", "compute-0,other"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.actions_context({**environment, field: value})

    def test_runner_temp_cannot_overlap_checkout(self):
        environment = self.actions_environment()
        for directory in (Path(environment["GITHUB_WORKSPACE"]), self.root):
            with self.subTest(directory=directory), self.assertRaisesRegex(
                ValueError, "separate"
            ):
                self.actions_context({**environment, "RUNNER_TEMP": str(directory)})

    def test_driver_cannot_run_from_a_different_checkout(self):
        environment = self.actions_environment()
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(client, "REPO", self.root / "different"),
            self.assertRaisesRegex(ValueError, "GITHUB_WORKSPACE checkout"),
        ):
            client.ActionsContext.from_environment()

    def test_legacy_actions_and_arbitrary_cli_options_are_rejected(self):
        for arguments in (
            ["status"],
            ["resume"],
            ["setup"],
            ["cleanup"],
            ["setup-ssh"],
            ["run", "--source-sha", "a" * 40],
            ["run", "--source-dir", str(self.root)],
            ["run", "--reuse-image-sha", "b" * 64],
            ["finalize", "--run-key", "other"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch.object(sys, "argv", ["slurm_client.py", *arguments]),
                patch.object(sys, "stderr", io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                client.main()
            self.assertEqual(error.exception.code, 2)

    def test_actions_run_builds_exact_checkout(self):
        source, sha = self.checkout(
            {"file": b"source", "scripts/ci/helper.py": b"# helper\n"}
        )
        context = self.actions_context(self.actions_environment(source, sha))
        remote_run = "/remote/runs/" + context.run_key

        class FakeConnection:
            deadline = client.time.monotonic() + client.RUN_TIMEOUT

            def call(self, _words, **kwargs):
                with tarfile.open(fileobj=kwargs["stdin"], mode="r:") as archive:
                    self.request = json.load(archive.extractfile("request.json"))
                return SimpleNamespace(
                    stdout=json.dumps({"run_dir": remote_run}).encode()
                )

        connection = FakeConnection()
        with (
            patch.object(client, "REPO", source),
            patch.object(client, "remote_action") as action,
            patch.object(client, "wait_for_run") as wait,
            patch.object(sys, "stdout", io.StringIO()),
        ):
            client.start_run(context, connection, {"root": "/remote"})
        self.assertEqual(connection.request["source_sha"], sha)
        self.assertEqual(connection.request["controller_sha"], sha)
        self.assertEqual(connection.request["run_key"], "gh-12345-2")
        self.assertEqual(
            client.read_json(context.output / "expected-request.json"),
            connection.request,
        )
        action.assert_called_once_with(connection, remote_run, "start")
        wait.assert_called_once_with(
            connection, remote_run, context.output, client.RUN_TIMEOUT
        )

    def test_finalize_without_submission_does_not_connect(self):
        context = self.actions_context()
        with patch.object(client, "preflight") as preflight:
            client.finalize_run(context, SimpleNamespace(deadline=None))
        preflight.assert_not_called()
        self.assertEqual(
            client.read_json(context.output / "finalize.json"),
            {"status": "not-submitted"},
        )

    def saved_actions_run(self, context):
        context.output.mkdir(parents=True)
        saved = {
            "run_key": context.run_key,
            "source_sha": context.sha,
            "remote_run": "/remote/runs/" + context.run_key,
            "ssh_host": "ci-slurm-login",
        }
        client.atomic_json(context.output / "client.json", saved)
        return saved

    def test_finalize_cancels_active_run_and_collects_even_on_failure(self):
        context = self.actions_context()
        saved = self.saved_actions_run(context)
        connection = SimpleNamespace(deadline=None)
        with (
            patch.object(client, "preflight", return_value={"root": "/remote"}),
            patch.object(
                client, "remote_action", side_effect=TimeoutError("disconnect")
            ) as action,
            patch.object(client, "collect") as collect,
            self.assertRaises(TimeoutError),
        ):
            client.finalize_run(context, connection)
        action.assert_called_once_with(connection, saved["remote_run"], "finalize", 180)
        collect.assert_called_once_with(
            connection, saved["remote_run"], context.output, timeout=60
        )
        self.assertIsNotNone(connection.deadline)

    def test_finalize_refuses_a_different_actions_attempt(self):
        context = self.actions_context()
        saved = self.saved_actions_run(context)
        saved["run_key"] = "gh-12345-1"
        client.atomic_json(context.output / "client.json", saved)
        with (
            patch.object(client, "preflight", return_value={"root": "/remote"}),
            patch.object(client, "remote_action") as action,
            self.assertRaisesRegex(ValueError, "Actions run identity"),
        ):
            client.finalize_run(context, SimpleNamespace(deadline=None))
        action.assert_not_called()

    def test_failed_terminal_states_never_pass(self):
        for state in ("FAILED", "CANCELLED", "TIMEOUT", "RUNNING", None):
            values = self.evidence()
            values["terminal"]["state"] = state
            client.atomic_json(self.root / "terminal.json", values["terminal"])
            with self.subTest(state=state), self.assertRaises(ValueError):
                client.verdict(self.root)

    def test_job_or_workload_identity_mismatch_never_passes(self):
        for file, field, value in (
            ("terminal", "job_id", "456"),
            ("wait-result", "job_id", "456"),
            ("completed", "source_sha", "0" * 40),
            ("completed", "run_key", "other"),
            ("completed", "controller_bundle_sha256", "0" * 64),
        ):
            values = self.evidence()
            values[file][field] = value
            client.atomic_json(self.root / (file + ".json"), values[file])
            with self.subTest(file=file, field=field), self.assertRaises(ValueError):
                client.verdict(self.root)

    def test_missing_record_zero_wait_never_passes(self):
        values = self.evidence()
        values["wait-result"]["missing_record"] = True
        client.atomic_json(self.root / "wait-result.json", values["wait-result"])
        with self.assertRaises(ValueError):
            client.verdict(self.root)

    def test_receipt_and_terminal_must_match_run(self):
        for file, field in (
            ("receipt", "run_key"),
            ("receipt", "source_sha"),
            ("terminal", "run_key"),
        ):
            values = self.evidence()
            values[file][field] = "other"
            client.atomic_json(self.root / (file + ".json"), values[file])
            with self.subTest(file=file, field=field), self.assertRaises(ValueError):
                client.verdict(self.root)

    def test_staging_transfer_checksum_failure_leaves_no_run(self):
        root = self.root / "dynamo-rocm-ci"
        with self.assertRaisesRegex(ValueError, "Transfer checksum mismatch"):
            remote.stage(root, "local-one", "0" * 64, io.BytesIO(b"not a tar archive"))
        self.assertFalse((root / "runs/local-one").exists())

    def test_existing_remote_run_never_overwritten(self):
        root = self.root / "dynamo-rocm-ci"
        target = root / "runs/local-one"
        target.mkdir(parents=True)
        (target / "marker").write_text("original")
        with self.assertRaisesRegex(ValueError, "Run already exists"):
            remote.stage(root, "local-one", "0" * 64, io.BytesIO())
        self.assertEqual((target / "marker").read_text(), "original")

    def package(self, name, content=b"content", symlink=False):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            member = tarfile.TarInfo(name)
            if symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
            else:
                member.size = len(content)
            archive.addfile(member, None if symlink else io.BytesIO(content))
        data = stream.getvalue()
        return io.BytesIO(data), hashlib.sha256(data).hexdigest()

    def test_remote_stage_checks_paths_and_entry_types(self):
        root = self.root / "dynamo-rocm-ci"
        for name, symlink in (
            ("../escape", False),
            ("unexpected", False),
            ("controller/link", True),
        ):
            stream, digest = self.package(name, symlink=symlink)
            with self.subTest(name=name), self.assertRaises(ValueError):
                remote.stage(root, "local-one", digest, stream)
            self.assertFalse((root / "runs/local-one").exists())

    def test_remote_stage_publishes_verified_bytes(self):
        root = self.root / "dynamo-rocm-ci"
        stream, digest = self.package("source.tar", b"verified source")
        result = remote.stage(root, "local-one", digest, stream)
        self.assertEqual(
            Path(result["run_dir"]).joinpath("source.tar").read_bytes(),
            b"verified source",
        )

    def test_remote_collection_excludes_non_evidence_and_links(self):
        (self.root / "request.json").write_text("{}")
        (self.root / "private.key").write_text("not evidence")
        (self.root / "leak.log").symlink_to(self.root / "private.key")
        (self.root / "cancel-request.json").write_text("{}")
        stream = io.BytesIO()
        remote.collect(self.root, stream)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            self.assertEqual(
                set(archive.getnames()),
                {"request.json", "cancel-request.json", "collection-report.json"},
            )

    def test_remote_failure_cannot_turn_into_verdict(self):
        with (
            patch.object(
                client, "remote_action", side_effect=TimeoutError("disconnected")
            ),
            patch.object(client, "verdict") as verdict,
        ):
            with self.assertRaises(TimeoutError):
                client.wait_for_run(
                    SimpleNamespace(deadline=None), "/run", self.root, 1
                )
            verdict.assert_not_called()

    def test_wait_does_not_reset_deadline_consumed_by_staging(self):
        connection = SimpleNamespace(deadline=10)
        with (
            patch.object(client.time, "monotonic", return_value=11),
            patch.object(client, "remote_action") as action,
            patch.object(client.time, "sleep") as sleep,
            self.assertRaisesRegex(TimeoutError, "Client deadline"),
        ):
            client.wait_for_run(connection, "/run", self.root, 17100)
        action.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(connection.deadline, 10)

    def test_retry_sleep_stays_inside_original_deadline(self):
        connection = SimpleNamespace(deadline=12)
        now = [10.0]

        def advance(seconds):
            now[0] += seconds

        with (
            patch.object(client.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(
                client, "remote_action", side_effect=TimeoutError("SSH timed out")
            ) as action,
            patch.object(client.time, "sleep", side_effect=advance) as sleep,
            self.assertRaisesRegex(TimeoutError, "Client deadline"),
        ):
            client.wait_for_run(connection, "/run", self.root, 17100)
        action.assert_called_once()
        sleep.assert_called_once_with(2)
        self.assertEqual(connection.deadline, 12)

    def test_collector_rejects_path_traversal(self):
        class FakeConnection:
            def call(self, _words, **kwargs):
                with tarfile.open(fileobj=kwargs["stdout"], mode="w") as archive:
                    item = tarfile.TarInfo("../escape")
                    item.size = 4
                    archive.addfile(item, io.BytesIO(b"oops"))

        with self.assertRaisesRegex(ValueError, "Unsafe evidence"):
            client.collect(FakeConnection(), "/run", self.root / "output")
        self.assertFalse((self.root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
