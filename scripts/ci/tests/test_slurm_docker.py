# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allocation confinement and private Docker resource ownership checks."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm.docker_daemon import DockerDaemon, mountpoints_below, slurm_job_cgroup


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


class DockerMountCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.daemon = DockerDaemon(self.root, self.root / "daemon.log")
        self.directory = self.root / "private"
        self.directory.mkdir()
        self.daemon.directory = self.directory
        self.daemon._directory_inode = self.directory.stat().st_ino

    @staticmethod
    def mountinfo(*paths):
        lines = []
        for index, path in enumerate(paths, 1):
            escaped = str(path).replace("\\", r"\134").replace(" ", r"\040")
            lines.append(f"{index} 0 0:1 / {escaped} rw - overlay overlay rw")
        return "\n".join(lines)

    @staticmethod
    def command_success(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=os.fsencode(command[-1]) + b"\0")

    def test_mount_filter_excludes_root_outside_and_prefix_collision(self):
        child = self.directory / "data" / "space in path"
        other = self.root / "private-other" / "data"
        table = self.mountinfo("/", self.root, self.directory, other, child)
        self.assertEqual(mountpoints_below(table, self.directory), [child])

    def test_nested_mounts_unmount_deepest_first_with_one_total_budget(self):
        parent = self.directory / "data" / "rootfs"
        child = parent / "proc"
        with (
            patch.object(
                Path, "read_text", side_effect=[self.mountinfo(parent, child), ""]
            ),
            patch("rocm.docker_daemon.time.monotonic", side_effect=range(100, 106)),
            patch(
                "rocm.docker_daemon.subprocess.run", side_effect=self.command_success
            ) as run,
        ):
            self.daemon._unmount_owned_filesystems()
        unmounts = [call for call in run.call_args_list if call.args[0][2] == "umount"]
        self.assertEqual(
            [call.args[0][-1] for call in unmounts], [str(child), str(parent)]
        )
        self.assertEqual([call.kwargs["timeout"] for call in unmounts], [13, 11])
        for call in unmounts:
            self.assertIn("--no-canonicalize", call.args[0])
            self.assertNotIn("--lazy", call.args[0])

    def test_no_mounts_only_removes_owned_directory_with_filesystem_backstop(self):
        with (
            patch.object(Path, "read_text", return_value=self.mountinfo("/")),
            patch("rocm.docker_daemon.subprocess.run") as run,
        ):
            self.daemon._cleanup()
        self.assertEqual(
            run.call_args.args[0],
            ["sudo", "-n", "rm", "-rf", "--one-file-system", "--", str(self.directory)],
        )
        self.assertEqual(run.call_count, 1)

    def test_unmount_failure_prevents_recursive_removal(self):
        mounted = self.directory / "data" / "rootfs"

        def fail_unmount(command, **kwargs):
            if command[2] == "umount":
                raise subprocess.CalledProcessError(32, command)
            return self.command_success(command, **kwargs)

        with (
            patch.object(Path, "read_text", return_value=self.mountinfo(mounted)),
            patch("rocm.docker_daemon.subprocess.run", side_effect=fail_unmount) as run,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            self.daemon._cleanup()
        self.assertEqual(
            [call.args[0][2] for call in run.call_args_list], ["realpath", "umount"]
        )

    def test_symlink_redirect_prevents_unmount_and_removal(self):
        mounted = self.directory / "data" / "rootfs"
        with (
            patch.object(Path, "read_text", return_value=self.mountinfo(mounted)),
            patch(
                "rocm.docker_daemon.subprocess.run",
                return_value=SimpleNamespace(stdout=b"/outside/rootfs\0"),
            ) as run,
            self.assertRaisesRegex(ValueError, "redirected"),
        ):
            self.daemon._cleanup()
        self.assertEqual([call.args[0][2] for call in run.call_args_list], ["realpath"])

    def test_lingering_mount_prevents_removal(self):
        mounted = self.directory / "data" / "rootfs"
        with (
            patch.object(Path, "read_text", return_value=self.mountinfo(mounted)),
            patch(
                "rocm.docker_daemon.subprocess.run", side_effect=self.command_success
            ) as run,
            self.assertRaisesRegex(RuntimeError, "mountpoints remain"),
        ):
            self.daemon._cleanup()
        self.assertNotIn("rm", [call.args[0][2] for call in run.call_args_list])

    def test_exhausted_total_deadline_prevents_more_commands(self):
        mounted = self.directory / "data" / "rootfs"
        with (
            patch.object(Path, "read_text", return_value=self.mountinfo(mounted)),
            patch("rocm.docker_daemon.time.monotonic", side_effect=[0, 1, 16]),
            patch(
                "rocm.docker_daemon.subprocess.run", side_effect=self.command_success
            ) as run,
            self.assertRaisesRegex(TimeoutError, "15 seconds"),
        ):
            self.daemon._cleanup()
        self.assertEqual([call.args[0][2] for call in run.call_args_list], ["realpath"])

    def test_replaced_directory_is_rechecked_before_unmount(self):
        mounted = self.directory / "data" / "rootfs"

        def replace_directory(command, **kwargs):
            self.directory.rename(self.root / "original")
            self.directory.mkdir()
            return self.command_success(command, **kwargs)

        with (
            patch.object(Path, "read_text", return_value=self.mountinfo(mounted)),
            patch(
                "rocm.docker_daemon.subprocess.run", side_effect=replace_directory
            ) as run,
            self.assertRaisesRegex(ValueError, "replaced"),
        ):
            self.daemon._cleanup()
        self.assertEqual([call.args[0][2] for call in run.call_args_list], ["realpath"])


if __name__ == "__main__":
    unittest.main()
