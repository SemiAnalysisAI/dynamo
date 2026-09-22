# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise batch-shell cancellation without submitting a Slurm job."""

import json
import os
import signal
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class BatchCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.scratch_temporary = tempfile.TemporaryDirectory(
            prefix="dynamo-rocm-test-", dir="/tmp"
        )
        self.addCleanup(self.scratch_temporary.cleanup)
        self.scratch = Path(self.scratch_temporary.name)
        (self.run / "request.json").write_text(
            json.dumps({"run_key": "cleanup-test", "source_sha": "a" * 40})
        )
        self.image = self.run / "dynamo-test.building.sqsh"
        self.image.write_bytes(b"partial image")
        batch = (Path(__file__).resolve().parents[1] / "slurm.sbatch").read_text()
        # Exercise the actual traps/functions, without the allocation probes
        # and GPU workload that follow them.
        self.prefix, separator, _ = batch.partition(': "${SLURM_JOB_ID:')
        self.assertTrue(separator)

    def run_shell(self, body, child=""):
        script = self.run / "batch.sh"
        script.write_text(self.prefix + "\nscratch=$2\n" + textwrap.dedent(body))
        (self.run / "child.py").write_text(textwrap.dedent(child))
        output = self.run / "output.log"
        with output.open("w") as stream:
            process = subprocess.Popen(
                ["bash", str(script), str(self.run), str(self.scratch)],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                result = process.wait(timeout=10)
            finally:
                # Also reap the deliberately unresponsive child in the
                # deadline case. Every process is in this test's own session.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        return result, output.read_text()

    def test_cancellation_reaps_child_before_removing_scratch(self):
        result, output = self.run_shell(
            """
            sleep() {
                : > "$RUN_DIR/reap-poll"
                command sleep "$@"
            }
            rm() {
                if [[ "${@: -1}" == "$scratch" && ! -f "$RUN_DIR/child-cleaned" ]]; then
                    echo 'scratch removed before child cleanup' >&2
                    return 9
                fi
                command rm "$@"
            }
            run_step python3 "$RUN_DIR/child.py" "$RUN_DIR" "$scratch"
            """,
            """
            import os, signal, sys, time
            from pathlib import Path
            run, scratch = map(Path, sys.argv[1:])
            (scratch / 'owned-storage').write_text('data')

            def terminate(signum, frame):
                # A handshake with the shell's first reap poll ensures that
                # this test detects deletion before waiting, without races.
                deadline = time.monotonic() + 5
                while not (run / 'reap-poll').exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('parent did not wait for child cleanup')
                    time.sleep(0.01)
                os.kill(os.getppid(), signal.SIGTERM)
                (scratch / 'owned-storage').unlink()
                (run / 'child-cleaned').touch()
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, terminate)
            os.kill(os.getppid(), signal.SIGTERM)
            signal.pause()
            """,
        )
        self.assertEqual(result, 143, output)
        self.assertTrue((self.run / "child-cleaned").exists(), output)
        self.assertFalse(self.scratch.exists(), output)
        self.assertFalse(self.image.exists(), output)
        self.assertIn("Slurm workload cleanup complete", output)
        self.assertEqual(
            json.loads((self.run / "completed.json").read_text())["exit_code"], 143
        )

    def test_scratch_failure_still_deletes_image_and_preserves_original_status(self):
        for status in (0, 7, 143):
            with self.subTest(status=status):
                self.image.write_bytes(b"partial image")
                result, output = self.run_shell(
                    """
                    rm() {
                        if [[ "${@: -1}" == "$scratch" ]]; then
                            return 9
                        fi
                        command rm "$@"
                    }
                    exit """
                    + str(status)
                    + "\n"
                )
                self.assertEqual(result, status or 1, output)
                self.assertTrue(self.scratch.exists())
                self.assertFalse(self.image.exists(), output)
                self.assertIn("Slurm workload cleanup incomplete", output)

    def test_child_deadline_keeps_live_scratch_and_still_deletes_image(self):
        self.assertEqual(self.prefix.count("SECONDS + 150"), 1)
        self.prefix = self.prefix.replace("SECONDS + 150", "SECONDS + 1")
        for stopped in (False, True):
            with self.subTest(stopped=stopped):
                self.image.write_bytes(b"partial image")
                result, output = self.run_shell(
                    f'run_step python3 "$RUN_DIR/child.py" {int(stopped)}',
                    """
                    import os, signal, sys
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    os.kill(os.getppid(), signal.SIGTERM)
                    if sys.argv[1] == '1':
                        os.kill(os.getpid(), signal.SIGSTOP)
                    while True:
                        signal.pause()
                    """,
                )
                self.assertEqual(result, 143, output)
                self.assertTrue(self.scratch.exists(), output)
                self.assertFalse(self.image.exists(), output)
                self.assertIn("child did not exit within the cleanup deadline", output)
                self.assertIn("Slurm workload cleanup incomplete", output)
                self.assertNotIn("Slurm workload cleanup complete", output)


if __name__ == "__main__":
    unittest.main()
