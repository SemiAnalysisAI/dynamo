# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise generated OpenSSH configuration without connecting to a host."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slurm_ssh as ssh


class SSHTests(unittest.TestCase):
    def test_directory_does_not_expand_openssh_tokens(self):
        for name in ("${HOME}", "%d", "line\nbreak"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ssh.quoted_path(Path("/tmp") / name)

    def test_generated_route_is_understood_by_openssh(self):
        jumps, login = ssh.ssh_route(
            {
                "login": {"hostname": "login.example", "user": "ci"},
                "jumps": [
                    {
                        "hostname": "gateway.example",
                        "user": "gateway",
                        "identity": "gateway",
                    },
                    {"hostname": "jump.example", "user": "ci", "port": 6068},
                ],
            }
        )
        with tempfile.TemporaryDirectory(prefix="ssh test ") as directory:
            root = Path(directory)
            config = root / "config"
            config.write_text(ssh.render_config(root, jumps, login))
            result = subprocess.run(
                ["ssh", "-F", str(config), "-G", "ci-slurm-login"],
                capture_output=True,
                text=True,
                check=True,
            )
            values = dict(line.split(" ", 1) for line in result.stdout.splitlines())
            self.assertEqual(values["hostname"], "login.example")
            self.assertEqual(values["proxyjump"], "ci-slurm-hop-0,ci-slurm-hop-1")
            self.assertEqual(values["identityagent"], "none")
            self.assertEqual(values["stricthostkeychecking"], "true")
            self.assertEqual(values["forwardagent"], "no")

    def test_site_fields_cannot_inject_ssh_options(self):
        for update in (
            {"hostname": "host\nProxyCommand evil"},
            {"user": "ci\nForwardAgent yes"},
            {"hostname": "host;evil"},
            {"hostname": "%h"},
            {"port": True},
            {"port": 65536},
            {"identity": "personal"},
            {"ProxyCommand": "evil"},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                ssh.ssh_route(
                    {"login": {"hostname": "login.example", "user": "ci", **update}}
                )

    def test_unneeded_gateway_key_is_not_required(self):
        values = {
            "SLURM_SSH_CONFIG": json.dumps(
                {"login": {"hostname": "login.example", "user": "ci"}}
            ),
            "SLURM_SSH_KEY": "fake test key",
            "SLURM_KNOWN_HOSTS": "fake host pins",
        }
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "ssh"
            with patch.dict(os.environ, values, clear=True), patch.object(
                subprocess, "run"
            ):
                ssh.setup_ssh(root)
            self.assertFalse((root / "gateway_key").exists())
            self.assertEqual((root / "ssh_key").stat().st_mode & 0o777, 0o600)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            ssh.cleanup_ssh(root)
            self.assertFalse(root.exists())

    def test_failed_key_validation_removes_temporary_credentials(self):
        values = {
            "SLURM_SSH_CONFIG": json.dumps(
                {"login": {"hostname": "login.example", "user": "ci"}}
            ),
            "SLURM_SSH_KEY": "fake test key",
            "SLURM_KNOWN_HOSTS": "fake host pins",
        }
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "ssh"
            with patch.dict(os.environ, values, clear=True), patch.object(
                subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "ssh-keygen"),
            ), self.assertRaises(subprocess.CalledProcessError):
                ssh.setup_ssh(root)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
