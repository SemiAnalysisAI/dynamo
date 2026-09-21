# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allocation confinement and private Docker resource ownership checks."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm.docker_daemon import DockerDaemon, slurm_job_cgroup


class DockerDaemonTests(unittest.TestCase):
    def test_cgroup_parent_preserves_exact_slurm_allocation(self):
        path = "/system.slice/slurmstepd.scope/job_981/step_batch/user/task_0"
        self.assertEqual(
            slurm_job_cgroup("0::" + path + "\n", "981"),
            "/system.slice/slurmstepd.scope/job_981",
        )

    def test_cgroup_parent_rejects_host_other_job_and_ambiguous_paths(self):
        for contents, job in (
            ("0::/\n", "981"),
            ("0::/system.slice/docker.service\n", "981"),
            ("0::/slurm/job_9810/step_batch\n", "981"),
            ("0::/slurm/job_981/job_981\n", "981"),
            ("0::/slurm/job_981/../other\n", "981"),
            ("0::/slurm/job_981\n0::/slurm/job_981\n", "981"),
            ("1:memory:/slurm/job_981\n", "981"),
            ("0::/slurm/job_981\n", "981/other"),
        ):
            with self.subTest(contents=contents, job=job), self.assertRaises(
                ValueError
            ):
                slurm_job_cgroup(contents, job)

    def test_commands_target_only_the_private_endpoint_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daemon = DockerDaemon(root, root / "daemon.log")
            with self.assertRaises(RuntimeError):
                daemon.command("info")
            daemon.directory = root / "private"
            daemon.config_dir = daemon.directory / "config"
            daemon.socket_path = daemon.directory / "docker.sock"
            self.assertEqual(
                daemon.command("info"),
                [
                    "sudo",
                    "-n",
                    "docker",
                    "--config",
                    str(daemon.config_dir),
                    "--host",
                    "unix://" + str(daemon.socket_path),
                    "info",
                ],
            )

    def test_cleanup_refuses_replaced_directory_without_running_sudo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            daemon = DockerDaemon(root, root / "daemon.log")
            daemon.directory = root / "private"
            daemon.directory.mkdir()
            daemon._directory_inode = daemon.directory.stat().st_ino
            daemon.directory.rename(root / "original")
            daemon.directory.symlink_to(root / "original", target_is_directory=True)
            with (
                patch("rocm.docker_daemon.subprocess.run") as run,
                self.assertRaisesRegex(ValueError, "replaced"),
            ):
                daemon._cleanup()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
