# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A private Docker/BuildKit daemon confined to the current Slurm allocation."""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath

BUILDX_VERSION = "v0.37.1"
BUILDX_SHA256 = "9447199cdb435f25880548343c128a4b6650e8891ee598905d8d29d39a8e359b"


def slurm_job_cgroup(contents: str, job_id: str) -> str:
    """Select the allocation's empty parent, not its populated task cgroup."""
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("A numeric SLURM_JOB_ID is required")
    unified = [line[3:] for line in contents.splitlines() if line.startswith("0::")]
    if len(unified) != 1:
        raise ValueError("The Docker builder requires unified cgroup v2")
    path = unified[0]
    if not path.startswith("/") or any(part in (".", "..") for part in path.split("/")):
        raise ValueError("Invalid Slurm cgroup path")
    parts = PurePosixPath(path).parts
    component = "job_" + job_id
    if parts.count(component) != 1:
        raise ValueError("Current process is outside the requested Slurm allocation")
    return str(PurePosixPath(*parts[: parts.index(component) + 1]))


class DockerDaemon:
    """Own a job-local daemon, CLI configuration, socket, and image store.

    The host daemon and its configuration are never used. The caller must pass
    ``cgroup_parent`` to BuildKit builds; ordinary Docker containers inherit it
    from this daemon. ``env`` also permits unprivileged Enroot dockerd imports.
    """

    def __init__(self, scratch: Path, log_path: Path):
        self.scratch = Path(scratch).resolve()
        self.log_path = Path(log_path)
        self.directory = None
        self.config_dir = None
        self.socket_path = None
        self.containerd_socket = None
        self.cgroup_parent = None
        self.env = None
        self._process = None
        self._containerd_process = None
        self._log = None
        self._pid = None
        self._directory_inode = None

    def command(self, *args: str) -> list[str]:
        if self.directory is None:
            raise RuntimeError("DockerDaemon has not been entered")
        return [
            "sudo",
            "-n",
            "docker",
            "--config",
            str(self.config_dir),
            "--host",
            "unix://" + str(self.socket_path),
            *args,
        ]

    def __enter__(self):
        if self.directory is not None:
            raise RuntimeError("DockerDaemon cannot be entered twice")
        if not self.scratch.is_dir() or self.scratch.stat().st_uid != os.getuid():
            raise ValueError(
                "Docker scratch must be an existing directory owned by this user"
            )
        for program in ("sudo", "docker", "dockerd", "containerd", "curl"):
            if shutil.which(program) is None:
                raise RuntimeError("Required builder command is missing: " + program)
        self.cgroup_parent = slurm_job_cgroup(
            Path("/proc/self/cgroup").read_text(), os.environ.get("SLURM_JOB_ID", "")
        )
        cgroup = Path("/sys/fs/cgroup") / self.cgroup_parent.lstrip("/")
        if (cgroup / "cgroup.procs").read_text().strip():
            raise ValueError(
                "Slurm allocation cgroup must not contain direct processes"
            )
        if (cgroup / "memory.max").read_text().strip() == "max":
            raise ValueError("Docker builds require a bounded Slurm memory allocation")
        self.directory = Path(tempfile.mkdtemp(prefix="docker-", dir=self.scratch))
        self._directory_inode = self.directory.stat().st_ino
        self.config_dir = self.directory / "config"
        self.socket_path = self.directory / "docker.sock"
        self.containerd_socket = self.directory / "containerd.sock"
        self.env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "DOCKER_CONTEXT",
                "DOCKER_TLS_VERIFY",
                "DOCKER_CERT_PATH",
                "DOCKER_API_VERSION",
            }
        }
        self.env.update(
            DOCKER_CONFIG=str(self.config_dir),
            DOCKER_HOST="unix://" + str(self.socket_path),
        )
        try:
            if len(os.fsencode(self.containerd_socket)) + len(".ttrpc") >= 108:
                raise ValueError("Docker scratch path is too long for a Unix socket")
            self.config_dir.mkdir()
            self._log = self.log_path.open("wb")
            self._install_buildx()
            self._start_containerd()
            daemon_config = self.directory / "daemon.json"
            daemon_config.write_text("{}\n")
            args = [
                "sudo",
                "-n",
                "dockerd",
                "--config-file=" + str(daemon_config),
                "--data-root=" + str(self.directory / "data"),
                "--exec-root=" + str(self.directory / "exec"),
                "--pidfile=" + str(self.directory / "docker.pid"),
                "--containerd=" + str(self.containerd_socket),
                "--containerd-namespace=" + self.directory.name,
                "--containerd-plugins-namespace=" + self.directory.name + "-plugins",
                "--host=unix://" + str(self.socket_path),
                "--bridge=none",
                "--iptables=false",
                "--ip6tables=false",
                "--ip-forward=false",
                "--ip-masq=false",
                "--userland-proxy=false",
                "--storage-driver=overlay2",
                "--exec-opt=native.cgroupdriver=cgroupfs",
                "--cgroup-parent=" + self.cgroup_parent,
            ]
            self._log.write((json.dumps(args) + "\n").encode())
            self._log.flush()
            self._process = subprocess.Popen(
                args, stdout=self._log, stderr=subprocess.STDOUT
            )
            self._wait_ready()
            return self
        except BaseException:
            self._cleanup()
            raise

    def _install_buildx(self):
        plugins = self.config_dir / "cli-plugins"
        plugins.mkdir()
        binary = plugins / "docker-buildx"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--max-time",
                "120",
                "--output",
                str(binary),
                f"https://github.com/docker/buildx/releases/download/{BUILDX_VERSION}/buildx-{BUILDX_VERSION}.linux-amd64",
            ],
            check=True,
            timeout=150,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        with binary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != BUILDX_SHA256:
            raise ValueError(
                "Downloaded Buildx executable does not match its pinned digest"
            )
        binary.chmod(0o755)

    def _start_containerd(self):
        config = self.directory / "containerd.toml"
        config.write_text(
            "version = 3\n"
            'disabled_plugins = ["io.containerd.cri.v1.images", '
            '"io.containerd.cri.v1.runtime", "io.containerd.grpc.v1.cri"]\n'
        )
        # Record the root process's PID before exec; all arguments remain argv
        # entries rather than interpolated shell source.
        args = [
            "sudo",
            "-n",
            "sh",
            "-c",
            'echo "$$" > "$1"; shift; exec "$@"',
            "containerd-launch",
            str(self.directory / "containerd.pid"),
            "containerd",
            "--config=" + str(config),
            "--root=" + str(self.directory / "containerd-root"),
            "--state=" + str(self.directory / "containerd-state"),
            "--address=" + str(self.containerd_socket),
        ]
        self._log.write((json.dumps(args) + "\n").encode())
        self._log.flush()
        self._containerd_process = subprocess.Popen(
            args, stdout=self._log, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._containerd_process.poll() is not None:
                raise RuntimeError(
                    "Private containerd exited; inspect " + str(self.log_path)
                )
            if self.containerd_socket.exists():
                self._verify_process(
                    self.directory / "containerd.pid",
                    "--root=" + str(self.directory / "containerd-root"),
                )
                return
            time.sleep(0.5)
        raise TimeoutError("Private containerd did not start within 30 seconds")

    def _verify_process(self, pidfile, marker):
        pid = int(pidfile.read_text().strip())
        if not self._owns_process(pid, marker):
            raise ValueError("Pidfile does not identify the private daemon")
        actual = Path(f"/proc/{pid}/cgroup").read_text()
        if slurm_job_cgroup(actual, os.environ["SLURM_JOB_ID"]) != self.cgroup_parent:
            raise ValueError("Private daemon escaped the Slurm allocation")
        return pid

    def _wait_ready(self):
        deadline = time.monotonic() + 60
        pidfile = self.directory / "docker.pid"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    "Private dockerd exited; inspect " + str(self.log_path)
                )
            if self.socket_path.exists() and pidfile.exists():
                self._pid = self._verify_process(
                    pidfile, "--data-root=" + str(self.directory / "data")
                )
                self._verify_process(
                    self.directory / "containerd.pid",
                    "--root=" + str(self.directory / "containerd-root"),
                )
                subprocess.run(
                    [
                        "sudo",
                        "-n",
                        "chown",
                        f"{os.getuid()}:{os.getgid()}",
                        str(self.socket_path),
                    ],
                    check=True,
                    timeout=5,
                )
                result = subprocess.run(
                    self.command(
                        "info", "--format", "{{.CgroupDriver}} {{.DockerRootDir}}"
                    ),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    if result.stdout.strip() != "cgroupfs " + str(
                        self.directory / "data"
                    ):
                        raise ValueError(
                            "Unexpected Docker daemon storage or cgroup driver"
                        )
                    return
            time.sleep(1)
        raise TimeoutError(
            "Private Docker daemon did not become ready within 60 seconds"
        )

    @staticmethod
    def _owns_process(pid, marker):
        if pid is None:
            return False
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            return False
        return os.fsencode(marker) in command

    def _signal_process(self, pid, marker, name):
        if self._owns_process(pid, marker):
            result = subprocess.run(
                ["sudo", "-n", "kill", "-" + name, str(pid)], timeout=5
            )
            if result.returncode and self._owns_process(pid, marker):
                raise RuntimeError("Could not stop private daemon")

    def _stop_process(self, process, pidfile, marker):
        if process is None:
            return
        pid = int(pidfile.read_text().strip()) if pidfile.exists() else None
        self._signal_process(pid, marker, "TERM")
        if process.poll() is None and pid is None:
            process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._signal_process(pid, marker, "KILL")
            process.kill()
            process.wait(timeout=5)

    def _cleanup(self):
        # Slurm and the batch shell can both signal this process. Defer those
        # signals until owned resources are gone, including on normal exit.
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}
        )
        try:
            self._cleanup_owned_resources()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def _cleanup_owned_resources(self):
        try:
            if self.directory is not None:
                try:
                    self._stop_process(
                        self._process,
                        self.directory / "docker.pid",
                        "--data-root=" + str(self.directory / "data"),
                    )
                finally:
                    self._stop_process(
                        self._containerd_process,
                        self.directory / "containerd.pid",
                        "--root=" + str(self.directory / "containerd-root"),
                    )
                if (
                    self.directory.is_symlink()
                    or self.directory.parent != self.scratch
                    or self.directory.stat().st_ino != self._directory_inode
                ):
                    raise ValueError(
                        "Refusing to remove a replaced Docker scratch directory"
                    )
                subprocess.run(
                    [
                        "sudo",
                        "-n",
                        "rm",
                        "-rf",
                        "--one-file-system",
                        "--",
                        str(self.directory),
                    ],
                    check=True,
                    timeout=60,
                )
                print("Private Docker cleanup complete", flush=True)
        finally:
            if self._log is not None:
                self._log.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self._cleanup()
