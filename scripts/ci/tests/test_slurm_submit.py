# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler fault cases without contacting Slurm or consuming GPU time."""

import getpass
import importlib.util
import io
import itertools
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location(
    "slurm_submit", Path(__file__).resolve().parents[1] / "slurm-submit.py"
)
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.req = {
            "run_key": "local-123",
            "source_sha": "a" * 40,
            "preferred_partition": "compute-1",
            "fallback_partition": "compute-0",
        }
        stdbuf = self.run / "stdbuf"
        stdbuf.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
        stdbuf.chmod(0o755)

    def record(self, state="RUNNING", owner=None, job_id="123"):
        return f"JobId={job_id} JobName=dynamo-local-123 Comment=local-123 UserId={owner or getpass.getuser()}(1000) JobState={state} ExitCode=0:0 Restarts=0"

    def finalize(self, deadline=30):
        with (
            patch.object(
                sys,
                "argv",
                [
                    "slurm-submit.py",
                    "finalize",
                    "--run-dir",
                    str(self.run),
                    "--deadline-seconds",
                    str(deadline),
                ],
            ),
            patch.object(submit, "load_request", return_value=self.req),
            redirect_stdout(io.StringIO()),
        ):
            submit.main()

    def test_finalization_before_monitor_prevents_submission(self):
        self.finalize()
        with (
            patch.object(submit, "load_request", return_value=self.req),
            patch.object(submit.subprocess, "Popen") as popen,
            patch.object(submit, "select_partition") as select,
        ):
            submit.monitor(self.run)
        popen.assert_not_called()
        select.assert_not_called()
        self.assertFalse((self.run / "submission-intent.json").exists())
        self.assertEqual(
            submit.read_json(self.run / "controller-result.json")["status"],
            "cancelled",
        )

    def test_finalization_during_selection_prevents_submission(self):
        def finalize_while_selecting(*_):
            self.finalize()
            return "compute-1"

        with (
            patch.object(submit, "load_request", return_value=self.req),
            patch.object(submit.subprocess, "Popen") as popen,
            patch.object(
                submit, "select_partition", side_effect=finalize_while_selecting
            ),
        ):
            submit.monitor(self.run)
        popen.assert_not_called()
        self.assertTrue((self.run / "cancel-request.json").exists())
        self.assertFalse((self.run / "submission-intent.json").exists())

    def test_finalization_waits_for_in_flight_job_then_cancels_exact_id(self):
        submit.atomic_json(self.run / "submission-intent.json", {"started_at": 1})
        with (
            patch.object(
                submit, "recover", side_effect=[None, {"job_id": "123"}]
            ) as recover,
            patch.object(submit, "cancel") as cancel,
            patch.object(submit, "inspect_job", return_value={"state": "CANCELLED"}),
            patch.object(submit.time, "sleep") as sleep,
        ):
            self.finalize()
        self.assertEqual(recover.call_count, 2)
        sleep.assert_called_once_with(2)
        self.assertEqual(cancel.call_args.args, (self.run.resolve(), self.req, "123"))
        self.assertTrue((self.run / "cancel-request.json").exists())

    def test_in_flight_deadline_keeps_cancellation_armed(self):
        submit.atomic_json(self.run / "submission-intent.json", {"started_at": 1})
        now = [10.0]
        with (
            patch.object(submit, "recover", return_value=None),
            patch.object(submit.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(
                submit.time,
                "sleep",
                side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds),
            ),
            self.assertRaisesRegex(TimeoutError, "deadline"),
        ):
            self.finalize(deadline=2)
        self.assertEqual(now[0], 12)
        self.assertTrue((self.run / "cancel-request.json").exists())
        self.assertFalse((self.run / "controller-result.json").exists())

    def test_monitor_honors_cancellation_after_process_starts(self):
        executable = self.run / "sbatch"
        executable.write_text("#!/bin/sh\nprintf '123\\n'\nexit 0\n")
        executable.chmod(0o755)
        request = {
            **self.req,
            "queue_timeout_seconds": 1800,
            "controller_timeout_seconds": 17100,
        }
        popen = submit.subprocess.Popen

        def start_then_cancel(*args, **kwargs):
            process = popen(*args, **kwargs)
            submit.atomic_json(self.run / "cancel-request.json", {"requested_at": 1})
            return process

        with (
            patch.dict(
                os.environ, {"PATH": str(self.run) + os.pathsep + os.environ["PATH"]}
            ),
            patch.object(submit, "load_request", return_value=request),
            patch.object(submit, "select_partition", return_value="compute-1"),
            patch.object(submit.subprocess, "Popen", side_effect=start_then_cancel),
            patch.object(submit, "recover", return_value=None),
            patch.object(submit, "cancel") as cancel,
            patch.object(submit, "inspect_job", return_value={"state": "CANCELLED"}),
            patch.object(submit.time, "time", side_effect=itertools.count(100, 6)),
        ):
            submit.monitor(self.run)
        cancel.assert_called_once_with(self.run, request, "123")

    def test_monitor_persists_unexpected_error_and_reraises(self):
        with (
            patch.object(submit, "load_request", return_value=self.req),
            patch.object(
                submit, "select_partition", side_effect=RuntimeError("failed query")
            ),
            patch.object(submit, "recover", return_value=None),
            self.assertRaisesRegex(RuntimeError, "failed query"),
        ):
            submit.monitor(self.run)
        self.assertEqual(
            submit.read_json(self.run / "controller-result.json"),
            {"status": "indeterminate", "reason": "failed query", "diagnostics": []},
        )

    def test_receipt_strictly_numeric(self):
        self.assertEqual(submit.parse_job_id("123;slurm\n"), "123")
        for value in (
            "",
            "0",
            "-1",
            "123 456",
            "Submitted batch job 123",
            "123;slurm;extra",
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

    def selection(self, outputs):
        responses = [
            SimpleNamespace(returncode=0, stdout=value, stderr="") for value in outputs
        ]
        with patch.object(submit, "command", side_effect=responses) as command:
            chosen = submit.select_partition(self.run, self.req)
        return chosen, command

    def test_prefers_healthy_idle_compute_one(self):
        node = "NodeName=node1 State=IDLE+DYNAMIC_NORM Partitions=compute-1 Gres=gpu:amd_instinct_mi300x_oam:8(S:0-1) CPUEfctv=128 CPUAlloc=0 RealMemory=1500000 AllocMem=0"
        chosen, command = self.selection(["PartitionName=compute-1 State=UP", node])
        self.assertEqual(chosen, "compute-1")
        self.assertEqual(command.call_count, 2)
        evidence = submit.read_json(self.run / "partition-selection.json")
        self.assertEqual(evidence["preferred_partition"], "compute-1")
        self.assertEqual(evidence["fallback_partition"], "compute-0")
        self.assertEqual(evidence["eligible_idle_nodes"], ["node1"])

    def test_custom_partition_policy_prefers_idle_then_falls_back(self):
        request = {
            **self.req,
            "preferred_partition": "mi300-priority",
            "fallback_partition": "mi300-shared",
        }
        node = "NodeName=node1 State=IDLE Partitions=mi300-priority Gres=gpu:8 CPUEfctv=128 CPUAlloc=0 RealMemory=1500000 AllocMem=0"
        for state, expected, outputs in (
            ("IDLE", "mi300-priority", []),
            ("MIXED", "mi300-shared", ["PartitionName=mi300-shared State=UP"]),
        ):
            with self.subTest(state=state):
                responses = [
                    SimpleNamespace(returncode=0, stdout=value, stderr="")
                    for value in (
                        "PartitionName=mi300-priority State=UP",
                        node.replace("State=IDLE", f"State={state}"),
                        *outputs,
                    )
                ]
                with patch.object(submit, "command", side_effect=responses):
                    self.assertEqual(
                        submit.select_partition(self.run, request), expected
                    )

    def test_partition_names_reject_options_lists_and_shell_syntax(self):
        for value in (None, "", "-other", "one,two", "one;two", "a" * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                submit.validate_partition(value)
        self.assertEqual(submit.validate_partition("pool-1.test_2"), "pool-1.test_2")

    def test_request_requires_distinct_preferred_and_fallback_partitions(self):
        request = {
            **self.req,
            "controller_sha": "b" * 40,
            "archive_sha256": "c" * 64,
            "controller_bundle_sha256": "d" * 64,
            "fallback_partition": "compute-1",
        }
        with (
            patch.object(submit, "read_json", return_value=request),
            self.assertRaisesRegex(ValueError, "must differ"),
        ):
            submit.load_request(self.run)
        del request["preferred_partition"]
        with (
            patch.object(submit, "read_json", return_value=request),
            self.assertRaises(KeyError),
        ):
            submit.load_request(self.run)

    def test_falls_back_for_busy_drained_small_or_gpu_less_nodes(self):
        original = "NodeName=node1 State=IDLE Partitions=compute-1 Gres=gpu:8 CPUEfctv=128 CPUAlloc=0 RealMemory=1500000 AllocMem=0"
        for node in (
            original.replace("State=IDLE", "State=MIXED"),
            original.replace("State=IDLE", "State=IDLE+DRAIN"),
            original.replace("Gres=gpu:8", "Gres=(null)"),
            original.replace("CPUEfctv=128", "CPUEfctv=8"),
            original.replace("RealMemory=1500000", "RealMemory=32768"),
        ):
            chosen, _ = self.selection(
                [
                    "PartitionName=compute-1 State=UP",
                    node,
                    "PartitionName=compute-0 State=UP",
                ],
            )
            self.assertEqual(chosen, "compute-0")

    def test_falls_back_from_down_partition(self):
        chosen, command = self.selection(
            ["PartitionName=compute-1 State=DOWN", "PartitionName=compute-0 State=UP"],
        )
        self.assertEqual(chosen, "compute-0")
        self.assertEqual(command.call_count, 2)

    def test_failed_selection_query_never_guesses_fallback(self):
        result = SimpleNamespace(
            returncode=1, stdout="", stderr="controller unavailable"
        )
        with patch.object(submit, "command", return_value=result) as command:
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                submit.select_partition(self.run, self.req)
            self.assertEqual(command.call_count, 1)
        self.assertNotIn(
            "chosen_partition", submit.read_json(self.run / "partition-selection.json")
        )

    def test_malformed_node_or_unavailable_fallback_fails(self):
        with self.assertRaisesRegex(ValueError, "node selection"):
            self.selection(["PartitionName=compute-1 State=UP", "garbage"])
        with self.assertRaisesRegex(RuntimeError, "not available"):
            self.selection(
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
