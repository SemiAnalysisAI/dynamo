#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard-library-only SSH bootstrap, staging and evidence collection.

The client sends this reviewed file with ``python3 -c`` before the controller
bundle exists remotely. Keep it independent of repository imports.
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import socket
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

EVIDENCE_FILES = {
    "request.json",
    "source-manifest.json",
    "build-manifest.json",
    "image-manifest.json",
    "image-build.json",
    "runtime-image-inspect.json",
    "test-image-inspect.json",
    "rendered-runtime.Dockerfile",
    "test-image.Dockerfile",
    "model-manifest.json",
    "aiter-cache.json",
    "allocation.json",
    "completed.json",
    "hip.json",
    "import-origins.json",
    "test-manifest.json",
    "test-summary.json",
    "contract.json",
    "provenance.json",
    "receipt.json",
    "terminal.json",
    "wait-result.json",
    "controller-result.json",
    "monitor.json",
    "heartbeat.json",
    "cancel.json",
    "cancel-request.json",
    "scheduler-latest.json",
    "submission-intent.json",
    "partition-selection.json",
    "collection.txt",
    "environment.json",
    "wheel-manifest.json",
    "imports-collection.json",
    "frontend-collection.json",
    "aggregate-collection.json",
    "frontend-listeners.json",
    "aggregate-listeners.json",
}


def preflight(root, expected_uid, image):
    if sys.version_info < (3, 12):
        raise ValueError("The Slurm login host requires Python 3.12 or newer")
    if os.getuid() != expected_uid:
        raise ValueError("Slurm UID differs from the configured artifact owner")
    required = ("python3", "sbatch", "scontrol", "squeue", "scancel", "stdbuf")
    missing = [name for name in required if not shutil.which(name)]
    if missing:
        raise ValueError("Missing commands: " + ", ".join(missing))
    if image:
        if not re.fullmatch(r"[0-9a-f]{64}", image):
            raise ValueError("Invalid image digest")
        location = root / "images" / image
        for name in ("image.sqsh", "image-manifest.json"):
            if not os.access(location / name, os.R_OK):
                raise ValueError("Local parity artifact not readable: " + name)
        manifest = json.loads((location / "image-manifest.json").read_text())
        if manifest.get("sqsh_sha256") != image:
            raise ValueError("Stored image identity mismatch")
    return {
        "uid": os.getuid(),
        "host": socket.gethostname(),
        "home": str(root.parent),
        "root": str(root),
        "image_readable": bool(image),
    }


def stage(root, key, digest, stream):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", key):
        raise ValueError("Invalid run key")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid transfer digest")
    root.mkdir(mode=0o700, exist_ok=True)
    runs = root / "runs"
    runs.mkdir(mode=0o700, exist_ok=True)
    target = runs / key
    if target.exists():
        raise ValueError("Run already exists; use status/resume with the recorded key")
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as temporary:
        temporary = Path(temporary)
        package = temporary / "package.tar"
        calculated = hashlib.sha256()
        with package.open("wb") as output:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                output.write(block)
                calculated.update(block)
        if calculated.hexdigest() != digest:
            raise ValueError("Transfer checksum mismatch")
        unpack = temporary / "run"
        unpack.mkdir(mode=0o700)
        with tarfile.open(package, "r:") as archive:
            seen = set()
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or member.name in seen
                ):
                    raise ValueError("Unsafe staging path")
                seen.add(member.name)
                if not (member.isfile() or member.isdir()) or member.mode & 0o7000:
                    raise ValueError("Unsafe staging entry")
                if path.parts[0] not in (
                    "request.json",
                    "source.tar",
                    "source-manifest.json",
                    "controller",
                ):
                    raise ValueError("Unexpected staging entry")
            archive.extractall(unpack, filter="data")
        os.rename(unpack, target)
    return {"run_dir": str(target)}


def collect(run, stream):
    cap = 64 * 1024 * 1024
    total = 0
    omitted = []
    with tarfile.open(fileobj=stream, mode="w|") as archive:
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run)
            if relative.parts[0] not in ("test-results", "service-logs") and not (
                len(relative.parts) == 1
                and (relative.name in EVIDENCE_FILES or relative.suffix == ".log")
            ):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            if size > cap or total + size > 512 * 1024 * 1024:
                omitted.append(str(relative))
                continue
            archive.add(path, arcname=str(relative), recursive=False)
            total += size
        data = json.dumps({"omitted_size_limit": omitted}).encode()
        info = tarfile.TarInfo("collection-report.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("expected_uid", type=int)
    check.add_argument("image")
    upload = commands.add_parser("stage")
    upload.add_argument("key")
    upload.add_argument("digest")
    download = commands.add_parser("collect")
    download.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path.home() / "dynamo-rocm-ci"
    if args.action == "preflight":
        print(json.dumps(preflight(root, args.expected_uid, args.image)))
    elif args.action == "stage":
        print(json.dumps(stage(root, args.key, args.digest, sys.stdin.buffer)))
    else:
        collect(args.run, sys.stdout.buffer)


if __name__ == "__main__":
    main()
