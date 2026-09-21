# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free primitives shared by the Slurm CI controllers."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def validate_sha(value: str, length: int = 40) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"Expected a lowercase {length}-character hexadecimal digest")
    return value


def validate_run_key(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value
    ):
        raise ValueError(
            "Invalid run key: use 1–96 letters, digits, underscores or hyphens"
        )
    return value


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value) -> None:
    """Replace one receipt atomically; flush before acknowledging it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def controller_manifest(root):
    """Describe a staged controller bundle, excluding its own manifest/bytecode."""
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            "__pycache__" in relative.parts
            or relative.as_posix() == "bundle-manifest.json"
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"Controller bundle contains a symlink: {relative}")
        if path.is_file():
            files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                    "mode": path.stat().st_mode & 0o777,
                }
            )
    if not files:
        raise ValueError("Empty controller bundle")
    return {"schema_version": 1, "files": files}


def verify_controller(run_dir):
    run_dir = Path(run_dir)
    request = read_json(run_dir / "request.json")
    expected = read_json(run_dir / "controller" / "bundle-manifest.json")
    validate_sha(request["controller_sha"])
    validate_sha(request["controller_bundle_sha256"], 64)
    if sha256_json(expected) != request["controller_bundle_sha256"]:
        raise ValueError("Controller bundle manifest digest mismatch")
    if controller_manifest(run_dir / "controller") != expected:
        raise ValueError(
            "Staged controller bytes or permissions differ from its manifest"
        )
    return expected
