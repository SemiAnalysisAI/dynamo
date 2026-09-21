# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a scoped SSH configuration from trusted deployment configuration."""

import json
import os
import re
import subprocess
from pathlib import Path

PRIVATE_FILES = ("gateway_key", "ssh_key", "known_hosts", "config")


def ssh_host(value):
    if not isinstance(value, dict) or set(value) - {
        "hostname",
        "user",
        "port",
        "identity",
    }:
        raise ValueError("SSH hosts accept hostname, user, port and identity only")
    hostname = value.get("hostname", "")
    user = value.get("user", "")
    port = value.get("port", 22)
    identity = value.get("identity", "cluster")
    if not isinstance(hostname, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.:-]*", hostname
    ):
        raise ValueError("Invalid SSH hostname")
    if not isinstance(user, str) or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", user
    ):
        raise ValueError("Invalid SSH user")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    if identity not in ("cluster", "gateway"):
        raise ValueError("SSH identity must be cluster or gateway")
    return {"hostname": hostname, "user": user, "port": port, "identity": identity}


def ssh_route(value):
    if (
        not isinstance(value, dict)
        or set(value) - {"login", "jumps"}
        or "login" not in value
    ):
        raise ValueError("SSH configuration requires login and optional jumps")
    jumps = value.get("jumps", [])
    if not isinstance(jumps, list) or len(jumps) > 4:
        raise ValueError("SSH configuration supports at most four jump hosts")
    return [ssh_host(host) for host in jumps], ssh_host(value["login"])


def quoted_path(path):
    text = str(path)
    if any(character in text for character in "\r\n\0%$"):
        raise ValueError("Unsupported character in SSH directory")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_config(directory, jumps, login):
    lines = [
        "Host *",
        "  BatchMode yes",
        "  IdentitiesOnly yes",
        "  IdentityAgent none",
        "  StrictHostKeyChecking yes",
        "  ForwardAgent no",
        "  GlobalKnownHostsFile /dev/null",
        f"  UserKnownHostsFile {quoted_path(directory / 'known_hosts')}",
        "  ConnectTimeout 20",
        "  ServerAliveInterval 30",
        "  ServerAliveCountMax 3",
    ]
    aliases = [f"ci-slurm-hop-{index}" for index in range(len(jumps))]
    for alias, host in zip([*aliases, "ci-slurm-login"], [*jumps, login]):
        key = "gateway_key" if host["identity"] == "gateway" else "ssh_key"
        lines.extend(
            [
                f"Host {alias}",
                f"  HostName {host['hostname']}",
                f"  User {host['user']}",
                f"  Port {host['port']}",
                f"  IdentityFile {quoted_path(directory / key)}",
            ]
        )
        if alias == "ci-slurm-login" and aliases:
            lines.append("  ProxyJump " + ",".join(aliases))
    return "\n".join(lines) + "\n"


def setup_ssh(directory):
    directory = Path(directory).resolve()
    jumps, login = ssh_route(json.loads(os.environ["SLURM_SSH_CONFIG"]))
    config = render_config(directory, jumps, login)
    values = {
        "ssh_key": os.environ["SLURM_SSH_KEY"],
        "known_hosts": os.environ["SLURM_KNOWN_HOSTS"],
    }
    if any(host["identity"] == "gateway" for host in [*jumps, login]):
        values["gateway_key"] = os.environ["SLURM_GATEWAY_SSH_KEY"]
    if any(not value.strip() for value in values.values()):
        raise ValueError("SSH keys and known-host pins must not be empty")
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        for name, data in {**values, "config": config}.items():
            descriptor = os.open(
                directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "w") as stream:
                stream.write(data.rstrip() + "\n")
        for name in values:
            command = ["ssh-keygen", "-l", "-f", str(directory / name)]
            if name != "known_hosts":
                command = ["ssh-keygen", "-y", "-P", "", "-f", str(directory / name)]
            subprocess.run(
                command,
                check=True,
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
    except (OSError, subprocess.SubprocessError):
        cleanup_ssh(directory)
        raise


def cleanup_ssh(directory):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Refusing symlink SSH directory")
    for name in PRIVATE_FILES:
        (directory / name).unlink(missing_ok=True)
    if directory.exists():
        directory.rmdir()
