#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage reviewed source, control one Slurm allocation, and collect its evidence."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from slurm_common import (
    atomic_json,
    controller_manifest,
    read_json,
    sha256_file,
    sha256_json,
    validate_run_key,
    validate_sha,
)
from slurm_ssh import cleanup_ssh, setup_ssh
from slurm_verify import verify_collected

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PYTHON = [
    "env",
    "-u",
    "PYTHONPATH",
    "-u",
    "PYTHONHOME",
    "-u",
    "PYTHONUSERBASE",
    "-u",
    "PYTHONSTARTUP",
    "PYTHONNOUSERSITE=1",
    "python3",
    "-B",
]


def run_checked(command, *, timeout=60, **kwargs):
    return subprocess.run(command, check=True, timeout=timeout, **kwargs)


def git(path, *args):
    return run_checked(["git", "-C", str(path), *args], capture_output=True).stdout


def clean_checkout(path, sha):
    validate_sha(sha)
    if git(path, "rev-parse", "HEAD").decode().strip() != sha:
        raise ValueError(f"Checkout HEAD does not match the requested SHA: {path}")
    if git(path, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError(f"Checkout must be clean, including untracked files: {path}")


def source_archive(source, destination, sha):
    """Archive tracked, materialized bytes identically on macOS and Linux."""
    clean_checkout(source, sha)
    entries = []
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for raw in sorted(git(source, "ls-files", "--stage", "-z").split(b"\0")):
            if not raw:
                continue
            header, name = raw.split(b"\t", 1)
            git_mode, _, stage = header.decode().split()
            name = name.decode("utf-8")
            path = source / name
            if stage != "0" or git_mode not in ("100644", "100755", "120000"):
                raise ValueError(f"Unresolved entry or unsupported submodule: {name}")
            if name.startswith("/") or ".." in PurePosixPath(name).parts:
                raise ValueError("Unsafe source path")
            info = tarfile.TarInfo(name)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            if git_mode == "120000":
                target = os.readlink(path)
                data = target.encode()
                info.type = tarfile.SYMTYPE
                info.linkname = target
                info.mode = 0o777
                archive.addfile(info)
                entry = {"type": "symlink", "linkname": target}
            else:
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Unexpected tracked file type: {name}")
                with path.open("rb") as stream:
                    if stream.read(256).startswith(
                        b"version https://git-lfs.github.com/spec/v1\n"
                    ):
                        raise ValueError(
                            f"Unmaterialized LFS pointer: {name}; run git lfs pull"
                        )
                    stream.seek(0)
                    info.size = path.stat().st_size
                    info.mode = 0o755 if git_mode == "100755" else 0o644
                    archive.addfile(info, stream)
                data = None
                entry = {"type": "file"}
            entry.update(
                path=name,
                size=len(data) if data is not None else info.size,
                mode=info.mode,
                sha256=hashlib.sha256(data).hexdigest()
                if data is not None
                else sha256_file(path),
            )
            entries.append(entry)
    clean_checkout(source, sha)
    return {
        "schema_version": 1,
        "source_sha": sha,
        "archive_sha256": sha256_file(destination),
        "files": entries,
    }


def stage_controller(destination):
    destination.mkdir()
    for name in git(REPO, "ls-files", "-z", "scripts/ci").decode().split("\0"):
        if not name:
            continue
        relative = Path(name).relative_to("scripts/ci")
        if "tests" in relative.parts or relative.suffix == ".md":
            continue
        source = REPO / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Unsupported controller entry: {name}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = controller_manifest(destination)
    atomic_json(destination / "bundle-manifest.json", manifest)
    return sha256_json(manifest)


class Connection:
    def __init__(self, args):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*", args.ssh_host):
            raise ValueError("Invalid SSH host alias")
        self.prefix = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
        ]
        if args.ssh_config:
            self.prefix += ["-F", str(Path(args.ssh_config).resolve())]
        self.prefix += [args.ssh_host]
        self.deadline = None

    def call(self, words, *, timeout=60, **kwargs):
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Controller operation deadline exhausted")
        return run_checked(
            [*self.prefix, shlex.join([str(w) for w in words])],
            timeout=timeout,
            **kwargs,
        )

    def python(self, code, *args, timeout=60):
        response = self.call(
            [*PYTHON, "-c", code, *args], timeout=timeout, capture_output=True
        )
        return json.loads(response.stdout)


def remote_program():
    """Send the same reviewed bootstrap helper used by offline tests."""
    return (HERE / "slurm_remote.py").read_text()


def preflight(connection, args):
    image = args.reuse_image_sha or ""
    if image:
        validate_sha(image, 64)
    return connection.python(
        remote_program(), "preflight", str(args.expected_uid), image
    )


def collect(connection, remote_run, output, timeout=120):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        connection.call(
            [*PYTHON, "-c", remote_program(), "collect", remote_run],
            timeout=timeout,
            stdout=archive,
            stderr=subprocess.PIPE,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as handle:
            for member in handle.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not member.isfile():
                    raise ValueError("Unsafe evidence archive")
                target = output / member.name
                if any(parent.is_symlink() for parent in [target, *target.parents]):
                    raise ValueError("Evidence destination contains a symlink")
            handle.extractall(output, filter="data")


def remote_action(connection, remote_run, action, deadline=240, cancel=False):
    words = [
        *PYTHON,
        str(PurePosixPath(remote_run) / "controller/slurm-submit.py"),
        action,
        "--run-dir",
        remote_run,
        "--deadline-seconds",
        str(deadline),
    ]
    if cancel:
        words.append("--cancel-if-active")
    return json.loads(
        connection.call(
            words, timeout=deadline if action == "finalize" else 60, capture_output=True
        ).stdout
    )


def verdict(output):
    request = read_json(output / "request.json")
    intended = output / "expected-request.json"
    if request != read_json(intended):
        raise ValueError("Collected request differs from the original local request")
    receipt = read_json(output / "receipt.json")
    terminal = read_json(output / "terminal.json")
    waiter = read_json(output / "wait-result.json")
    controller = read_json(output / "controller-result.json")
    workload = read_json(output / "completed.json")
    expected = str(receipt["job_id"])
    if not re.fullmatch(r"[1-9][0-9]*", expected):
        raise ValueError("Invalid scheduler job ID")
    if (
        receipt.get("run_key") != request["run_key"]
        or receipt.get("source_sha") != request["source_sha"]
        or terminal.get("run_key") != request["run_key"]
    ):
        raise ValueError("Scheduler receipt belongs to a different run")
    if (
        controller.get("status") != "terminal"
        or terminal.get("state") != "COMPLETED"
        or terminal.get("exit_code") != "0:0"
        or terminal.get("restarts") != 0
        or str(terminal.get("job_id")) != expected
        or waiter.get("returncode") != 0
        or waiter.get("missing_record") is not False
        or str(waiter.get("job_id")) != expected
        or workload.get("status") != "passed"
        or str(workload.get("job_id")) != expected
    ):
        raise ValueError(
            "Scheduler/waiter/workload evidence does not establish success"
        )
    for key in (
        "run_key",
        "source_sha",
        "archive_sha256",
        "controller_sha",
        "controller_bundle_sha256",
    ):
        if workload.get(key) != request[key]:
            raise ValueError(f"Workload verdict identity mismatch: {key}")
    image = read_json(output / "image-manifest.json")
    validate_sha(image["sqsh_sha256"], 64)
    if workload.get("sqsh_sha256") != image["sqsh_sha256"]:
        raise ValueError("Workload image identity mismatch")
    if (
        request.get("reuse_image_sha")
        and request["reuse_image_sha"] != image["sqsh_sha256"]
    ):
        raise ValueError("Reused image differs from requested digest")
    verify_collected(request, output)
    return {"status": "passed", "job_id": expected, "run_key": request["run_key"]}


def wait_for_run(connection, remote_run, output, timeout):
    deadline = time.monotonic() + timeout
    if connection.deadline is not None:
        deadline = min(deadline, connection.deadline)
    # One deadline bounds retries, sleeps, status calls, and final collection.
    connection.deadline = deadline
    previous = None
    last_collection = time.monotonic()
    while time.monotonic() < deadline:
        try:
            state = remote_action(connection, remote_run, "status")
        except (OSError, subprocess.SubprocessError) as error:
            atomic_json(
                output / "connection-error.json",
                {"error": str(error), "at": time.time()},
            )
            time.sleep(min(20, max(0, deadline - time.monotonic())))
            continue
        atomic_json(output / "client-status.json", state)
        display = (state["phase"], state.get("receipt", {}).get("job_id"))
        if display != previous:
            print(json.dumps(state), flush=True)
            previous = display
        if not state.get("active"):
            collect(connection, remote_run, output)
            result = verdict(output)
            atomic_json(output / "client-result.json", result)
            print(json.dumps(result), flush=True)
            return
        if time.monotonic() - last_collection >= 120:
            try:
                collect(connection, remote_run, output, timeout=45)
            except (OSError, subprocess.SubprocessError) as error:
                atomic_json(
                    output / "collection-error.json",
                    {"error": str(error), "at": time.time()},
                )
            last_collection = time.monotonic()
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    raise TimeoutError("Client deadline reached; use finalize/status with this run key")


def start_run(args, connection, info):
    source = Path(args.source_dir).resolve()
    output = Path(args.output).resolve()
    clean_checkout(REPO, args.controller_sha)
    clean_checkout(source, args.source_sha)
    if output.is_relative_to(REPO) or output.is_relative_to(source):
        raise ValueError("Evidence must be outside both clean checkouts")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run-key.txt").write_text(args.run_key + "\n")
    remote_run = str(PurePosixPath(info["root"]) / "runs" / args.run_key)
    atomic_json(
        output / "client.json",
        {
            "run_key": args.run_key,
            "remote_run": remote_run,
            "ssh_host": args.ssh_host,
            "started_at": time.time(),
            "deadline_at": time.time() + max(0, connection.deadline - time.monotonic()),
        },
    )
    print(f"Run {args.run_key}; evidence {output}; remote {remote_run}", flush=True)
    with tempfile.TemporaryDirectory(prefix="slurm-source-") as scratch:
        stage = Path(scratch) / "run"
        stage.mkdir()
        source_manifest = source_archive(source, stage / "source.tar", args.source_sha)
        atomic_json(stage / "source-manifest.json", source_manifest)
        bundle_sha = stage_controller(stage / "controller")
        request = {
            "schema_version": 1,
            "run_key": args.run_key,
            "source_sha": args.source_sha,
            "archive_sha256": source_manifest["archive_sha256"],
            "controller_sha": args.controller_sha,
            "controller_bundle_sha256": bundle_sha,
            "reuse_image_sha": args.reuse_image_sha,
            "suite": "aggregate",
            "partition": args.partition,
            "preferred_partition": args.preferred_partition,
            "fallback_partition": args.fallback_partition,
            "gpus": 1,
            "cpus": 16,
            "mem_gib": 64,
            "time_limit_minutes": 240,
            "queue_timeout_seconds": 1800,
            "controller_timeout_seconds": 17100,
            "expected_uid": args.expected_uid,
        }
        atomic_json(stage / "request.json", request)
        atomic_json(output / "request.json", request)
        atomic_json(output / "expected-request.json", request)
        package = Path(scratch) / "package.tar"
        with tarfile.open(package, "w") as archive:
            for path in sorted(stage.rglob("*")):
                archive.add(path, arcname=path.relative_to(stage), recursive=False)
        with package.open("rb") as stream:
            response = connection.call(
                [
                    *PYTHON,
                    "-c",
                    remote_program(),
                    "stage",
                    args.run_key,
                    sha256_file(package),
                ],
                timeout=900,
                stdin=stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        if json.loads(response.stdout)["run_dir"] != remote_run:
            raise ValueError("Remote staging path mismatch")
    remote_action(connection, remote_run, "start")
    wait_for_run(connection, remote_run, output, 17100)


def main():
    if sys.version_info < (3, 12):
        raise ValueError("The Slurm client requires Python 3.12 or newer")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "setup-ssh",
            "cleanup-ssh",
            "preflight",
            "run",
            "status",
            "resume",
            "finalize",
        ),
    )
    parser.add_argument("--directory")
    parser.add_argument("--ssh-host", default="slurm-login")
    parser.add_argument("--ssh-config")
    parser.add_argument("--expected-uid", type=int)
    parser.add_argument("--run-key")
    parser.add_argument("--output")
    parser.add_argument("--source-dir")
    parser.add_argument("--source-sha")
    parser.add_argument("--controller-sha")
    parser.add_argument("--reuse-image-sha")
    parser.add_argument("--suite", choices=("aggregate",), default="aggregate")
    parser.add_argument("--partition", default="auto")
    parser.add_argument("--preferred-partition", default="compute-1")
    parser.add_argument("--fallback-partition", default="compute-0")
    parser.add_argument("--gpus", type=int, choices=(1,), default=1)
    parser.add_argument("--queue-timeout", choices=("30m",), default="30m")
    parser.add_argument("--time-limit", choices=("04:00:00",), default="04:00:00")
    parser.add_argument("--controller-timeout", choices=("285m",), default="285m")
    parser.add_argument("--deadline-seconds", type=int, default=240)
    parser.add_argument("--cancel-if-active", action="store_true")
    args = parser.parse_args()
    if args.action in ("setup-ssh", "cleanup-ssh"):
        if not args.directory:
            parser.error("--directory is required")
        (setup_ssh if args.action == "setup-ssh" else cleanup_ssh)(args.directory)
        return
    if args.expected_uid is None or args.expected_uid < 1:
        parser.error(
            "--expected-uid must identify the configured non-root Slurm account"
        )
    for name in (args.partition, args.preferred_partition, args.fallback_partition):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            parser.error("invalid partition name")
    if (
        "auto" in (args.preferred_partition, args.fallback_partition)
        or args.preferred_partition == args.fallback_partition
    ):
        parser.error(
            "preferred and fallback partitions must be distinct explicit names"
        )
    connection = Connection(args)
    if args.action == "preflight":
        print(json.dumps(preflight(connection, args)))
        return
    validate_run_key(args.run_key)
    if not args.output:
        parser.error("--output is required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "finalize" and not (output / "client.json").exists():
        atomic_json(output / "finalize.json", {"status": "not-submitted"})
        return
    if args.action == "run":
        connection.deadline = time.monotonic() + 17100
    elif args.action == "finalize":
        connection.deadline = time.monotonic() + args.deadline_seconds
    elif args.action == "resume":
        saved = read_json(output / "client.json")
        expires = saved.get("deadline_at", saved["started_at"] + 17100)
        connection.deadline = time.monotonic() + max(0, expires - time.time())
    info = preflight(connection, args)
    remote_run = str(PurePosixPath(info["root"]) / "runs" / args.run_key)
    if args.action == "run":
        if not all((args.source_dir, args.source_sha, args.controller_sha)):
            parser.error("run requires --source-dir, --source-sha and --controller-sha")
        start_run(args, connection, info)
        return
    if not 1 <= args.deadline_seconds <= 240:
        parser.error("--deadline-seconds must be 1..240")
    try:
        remote_budget = (
            min(args.deadline_seconds, 180)
            if args.action == "finalize"
            else args.deadline_seconds
        )
        state = remote_action(
            connection, remote_run, args.action, remote_budget, args.cancel_if_active
        )
        print(json.dumps(state), flush=True)
        atomic_json(output / "client-status.json", state)
    finally:
        if args.action == "finalize":
            collect(connection, remote_run, output, timeout=60)
    if args.action == "resume":
        remaining = max(0, connection.deadline - time.monotonic())
        wait_for_run(connection, remote_run, output, remaining)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Slurm CI: {error}", file=sys.stderr)
        if error.stderr:
            detail = error.stderr
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            print(detail[-8192:], file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"Slurm CI: {error}", file=sys.stderr)
        sys.exit(1)
