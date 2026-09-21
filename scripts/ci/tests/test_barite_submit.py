# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler fault cases without contacting Slurm or consuming GPU time."""

import getpass
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location(
    "barite_submit", Path(__file__).resolve().parents[1] / "barite-submit.py"
)
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.req = {"run_key": "local-123", "source_sha": "a" * 40}
        stdbuf = self.run / "stdbuf"
        stdbuf.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
        stdbuf.chmod(0o755)

    def record(self, state="RUNNING", owner=None, job_id="123"):
        return f"JobId={job_id} JobName=dynamo-local-123 Comment=local-123 UserId={owner or getpass.getuser()}(1000) JobState={state} ExitCode=0:0 Restarts=0"

    def test_receipt_strictly_numeric(self):
        self.assertEqual(submit.parse_job_id("123;barite\n"), "123")
        for value in (
            "",
            "0",
            "-1",
            "123 456",
            "Submitted batch job 123",
            "123;barite;extra",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                submit.parse_job_id(value)

    def test_missing_record_does_not_persist_success(self):
        with patch.object(
            submit, "command", return_value=SimpleNamespace(returncode=1, stdout="")
        ):
            self.assertIsNone(submit.inspect_job(self.run, self.req, "123"))
        self.assertFalse((self.run / "terminal.json").exists())

    def test_terminal_requires_identity(self):
        with (
            patch.object(
                submit,
                "command",
                return_value=SimpleNamespace(
                    returncode=0, stdout=self.record("COMPLETED", "another-user")
                ),
            ),
            self.assertRaises(ValueError),
        ):
            submit.inspect_job(self.run, self.req, "123")
        self.assertFalse((self.run / "terminal.json").exists())

    def test_completed_evidence_persisted(self):
        with patch.object(
            submit,
            "command",
            return_value=SimpleNamespace(returncode=0, stdout=self.record("COMPLETED")),
        ):
            evidence = submit.inspect_job(self.run, self.req, "123")
        self.assertEqual(evidence["state"], "COMPLETED")
        self.assertEqual(
            submit.read_json(self.run / "terminal.json")["exit_code"], "0:0"
        )

    def test_cancel_only_exact_verified_job(self):
        with patch.object(
            submit,
            "command",
            side_effect=[
                SimpleNamespace(returncode=0, stdout=self.record()),
                SimpleNamespace(returncode=0),
            ],
        ) as command:
            submit.cancel(self.run, self.req, "123")
        self.assertEqual(command.call_args_list[-1].args[0], ["scancel", "123"])

    def test_cancel_missing_or_wrong_job_never_calls_scancel(self):
        for result in (
            SimpleNamespace(returncode=1, stdout=""),
            SimpleNamespace(returncode=0, stdout=self.record(job_id="456")),
        ):
            with (
                self.subTest(result=result),
                patch.object(submit, "command", return_value=result) as command,
            ):
                with self.assertRaises((RuntimeError, ValueError)):
                    submit.cancel(self.run, self.req, "123")
                self.assertEqual(command.call_count, 1)

    def test_duplicate_recovery_fails_closed(self):
        with (
            patch.object(
                submit,
                "command",
                return_value=SimpleNamespace(returncode=0, stdout="123\n456\n"),
            ),
            patch.object(submit, "inspect_job", return_value={"state": "RUNNING"}),
            self.assertRaisesRegex(RuntimeError, "multiple"),
        ):
            submit.recover(self.run, self.req)
        self.assertFalse((self.run / "receipt.json").exists())

    def test_zero_wait_without_terminal_is_indeterminate(self):
        executable = self.run / "sbatch"
        executable.write_text("#!/bin/sh\nprintf '123\\n'\nexit 0\n")
        executable.chmod(0o755)
        request = {
            **self.req,
            "queue_timeout_seconds": 1800,
            "controller_timeout_seconds": 17100,
        }
        with (
            patch.dict(
                os.environ, {"PATH": str(self.run) + os.pathsep + os.environ["PATH"]}
            ),
            patch.object(submit, "load_request", return_value=request),
            patch.object(submit, "select_partition", return_value="compute-0"),
            patch.object(submit, "inspect_job", return_value=None),
            patch.object(submit, "settle_terminal", return_value=None),
        ):
            submit.monitor(self.run)
        self.assertEqual(
            submit.read_json(self.run / "wait-result.json")["returncode"], 0
        )
        self.assertEqual(
            submit.read_json(self.run / "controller-result.json")["status"],
            "indeterminate",
        )

    def test_missing_record_diagnostic_is_retained_even_with_zero_exit(self):
        executable = self.run / "sbatch"
        executable.write_text(
            "#!/bin/sh\nprintf '123\\n'\nprintf 'job no longer found and exit code not found\\n' >&2\nexit 0\n"
        )
        executable.chmod(0o755)
        request = {
            **self.req,
            "queue_timeout_seconds": 1800,
            "controller_timeout_seconds": 17100,
        }
        with (
            patch.dict(
                os.environ, {"PATH": str(self.run) + os.pathsep + os.environ["PATH"]}
            ),
            patch.object(submit, "load_request", return_value=request),
            patch.object(submit, "select_partition", return_value="compute-0"),
            patch.object(submit, "inspect_job", return_value={"state": "COMPLETED"}),
        ):
            submit.monitor(self.run)
        self.assertTrue(
            submit.read_json(self.run / "wait-result.json")["missing_record"]
        )

    def test_terminal_before_waiter_keeps_monitor_active(self):
        submit.atomic_json(self.run / "terminal.json", {"state": "COMPLETED"})
        state = submit.snapshot(self.run)
        self.assertTrue(state["active"])
        self.assertEqual(state["phase"], "awaiting-waiter")

    def test_waiter_exit_settles_completing_to_cancelled(self):
        now = [10.0]
        records = [self.record("COMPLETING"), self.record("CANCELLED")]
        with (
            patch.object(submit.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(
                submit.time,
                "sleep",
                side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds),
            ),
            patch.object(
                submit,
                "command",
                side_effect=[
                    SimpleNamespace(returncode=0, stdout=value) for value in records
                ],
            ),
        ):
            result = submit.settle_terminal(self.run, self.req, "123", 20)
        self.assertEqual(result["state"], "CANCELLED")
        self.assertEqual(
            submit.read_json(self.run / "terminal.json")["state"], "CANCELLED"
        )
        self.assertTrue((self.run / "heartbeat.json").exists())

    def test_missing_terminal_settle_stops_at_deadline(self):
        now = [10.0]
        with (
            patch.object(submit.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(
                submit.time,
                "sleep",
                side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds),
            ),
            patch.object(submit, "inspect_job", return_value=None) as inspect,
        ):
            result = submit.settle_terminal(self.run, self.req, "123", 13)
        self.assertIsNone(result)
        self.assertEqual(now[0], 13)
        self.assertEqual(inspect.call_count, 2)
        self.assertFalse((self.run / "terminal.json").exists())

    def test_late_terminal_reconciles_only_missing_evidence_result(self):
        submit.atomic_json(self.run / "receipt.json", {**self.req, "job_id": "123"})
        submit.atomic_json(
            self.run / "terminal.json",
            {"run_key": self.req["run_key"], "job_id": "123", "state": "CANCELLED"},
        )
        submit.atomic_json(
            self.run / "wait-result.json",
            {"job_id": "123", "returncode": 143, "missing_record": False},
        )
        for reason, expected in (
            (submit.WAIT_FINISHED_REASON, "terminal"),
            ("scheduler identity mismatch", "indeterminate"),
        ):
            submit.atomic_json(
                self.run / "controller-result.json",
                {"status": "indeterminate", "reason": reason},
            )
            submit.reconcile_terminal(self.run, self.req)
            self.assertEqual(
                submit.read_json(self.run / "controller-result.json")["status"],
                expected,
            )

    def test_late_terminal_different_job_never_reconciles(self):
        submit.atomic_json(self.run / "receipt.json", {**self.req, "job_id": "123"})
        submit.atomic_json(
            self.run / "terminal.json",
            {"run_key": self.req["run_key"], "job_id": "456", "state": "CANCELLED"},
        )
        submit.atomic_json(self.run / "wait-result.json", {"job_id": "123"})
        result = {"status": "indeterminate", "reason": submit.WAIT_FINISHED_REASON}
        submit.atomic_json(self.run / "controller-result.json", result)
        submit.reconcile_terminal(self.run, self.req)
        self.assertEqual(submit.read_json(self.run / "controller-result.json"), result)

    def selection(self, policy, outputs):
        responses = [
            SimpleNamespace(returncode=0, stdout=value, stderr="") for value in outputs
        ]
        with patch.object(submit, "command", side_effect=responses) as command:
            chosen = submit.select_partition(
                self.run, {**self.req, "partition": policy}
            )
        return chosen, command

    def test_auto_prefers_healthy_idle_compute_one(self):
        node = "NodeName=node1 State=IDLE+DYNAMIC_NORM Partitions=compute-1 Gres=gpu:amd_instinct_mi300x_oam:8(S:0-1) CPUEfctv=128 CPUAlloc=0 RealMemory=1500000 AllocMem=0"
        chosen, command = self.selection(
            "auto", ["PartitionName=compute-1 State=UP", node]
        )
        self.assertEqual(chosen, "compute-1")
        self.assertEqual(command.call_count, 2)
        evidence = submit.read_json(self.run / "partition-selection.json")
        self.assertEqual(evidence["requested_policy"], "auto")
        self.assertEqual(evidence["eligible_idle_nodes"], ["node1"])

    def test_auto_falls_back_for_busy_drained_small_or_gpu_less_nodes(self):
        original = "NodeName=node1 State=IDLE Partitions=compute-1 Gres=gpu:8 CPUEfctv=128 CPUAlloc=0 RealMemory=1500000 AllocMem=0"
        for node in (
            original.replace("State=IDLE", "State=MIXED"),
            original.replace("State=IDLE", "State=IDLE+DRAIN"),
            original.replace("Gres=gpu:8", "Gres=(null)"),
            original.replace("CPUEfctv=128", "CPUEfctv=8"),
            original.replace("RealMemory=1500000", "RealMemory=32768"),
        ):
            chosen, _ = self.selection(
                "auto",
                [
                    "PartitionName=compute-1 State=UP",
                    node,
                    "PartitionName=compute-0 State=UP",
                ],
            )
            self.assertEqual(chosen, "compute-0")

    def test_auto_falls_back_from_down_partition(self):
        chosen, command = self.selection(
            "auto",
            ["PartitionName=compute-1 State=DOWN", "PartitionName=compute-0 State=UP"],
        )
        self.assertEqual(chosen, "compute-0")
        self.assertEqual(command.call_count, 2)

    def test_explicit_partition_override_is_preserved(self):
        for partition in ("compute-0", "compute-1"):
            chosen, command = self.selection(
                partition, [f"PartitionName={partition} State=UP"]
            )
            self.assertEqual(chosen, partition)
            self.assertEqual(command.call_count, 1)

    def test_failed_selection_query_never_guesses_fallback(self):
        result = SimpleNamespace(
            returncode=1, stdout="", stderr="controller unavailable"
        )
        with patch.object(submit, "command", return_value=result) as command:
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                submit.select_partition(self.run, {**self.req, "partition": "auto"})
            self.assertEqual(command.call_count, 1)
        self.assertNotIn(
            "chosen_partition", submit.read_json(self.run / "partition-selection.json")
        )

    def test_malformed_node_or_unavailable_fallback_fails(self):
        with self.assertRaisesRegex(ValueError, "node selection"):
            self.selection("auto", ["PartitionName=compute-1 State=UP", "garbage"])
        with self.assertRaisesRegex(RuntimeError, "not available"):
            self.selection(
                "auto",
                [
                    "PartitionName=compute-1 State=DOWN",
                    "PartitionName=compute-0 State=DOWN",
                ],
            )

    def test_unique_recovery_keeps_identity(self):
        with (
            patch.object(
                submit,
                "command",
                return_value=SimpleNamespace(returncode=0, stdout="123\n"),
            ),
            patch.object(submit, "inspect_job", return_value={"state": "RUNNING"}),
        ):
            receipt = submit.recover(self.run, self.req)
        self.assertEqual(receipt["job_id"], "123")
        self.assertTrue(receipt["recovered"])


if __name__ == "__main__":
    unittest.main()
