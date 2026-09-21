#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish one immutable, content-addressed Hugging Face home under a lock."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(home):
    records = {}
    for path in sorted(home.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(home.resolve()):
                raise ValueError(f"Escaping cache symlink: {path}")
            records[str(path.relative_to(home))] = digest(path)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name("contract.json").read_text())
    model = contract["model"]
    args.cache_root.mkdir(parents=True, exist_ok=True)
    with (args.cache_root / ".publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Reuse only a fully rehashed immutable cache for this exact revision.
        for candidate in args.cache_root.iterdir():
            manifest = candidate / "model-content.json"
            if not candidate.is_dir() or not manifest.is_file():
                continue
            data = json.loads(manifest.read_text())
            if data.get("model") != model:
                continue
            expected = data["files"]
            actual = inventory(candidate)
            actual.pop("model-content.json", None)
            if (
                actual != expected
                or hashlib.sha256(manifest.read_bytes()).hexdigest() != candidate.name
            ):
                raise ValueError("Published model cache failed content verification")
            publish_result(args.output, candidate)
            return

        staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=args.cache_root))
        try:
            snapshot_download(
                repo_id=model["repo_id"],
                revision=model["revision"],
                cache_dir=staging / "hub",
            )
            repo = staging / "hub" / ("models--" + model["repo_id"].replace("/", "--"))
            (repo / "refs").mkdir(exist_ok=True)
            (repo / "refs/main").write_text(model["revision"])
            shutil.rmtree(staging / "hub/.locks", ignore_errors=True)
            data = {"model": model, "files": inventory(staging)}
            raw = (
                json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            key = hashlib.sha256(raw).hexdigest()
            (staging / "model-content.json").write_bytes(raw)
            for path in staging.rglob("*"):
                if not path.is_symlink():
                    path.chmod(0o555 if path.is_dir() else 0o444)
            staging.chmod(0o555)
            destination = args.cache_root / key
            os.rename(staging, destination)
            publish_result(args.output, destination)
        finally:
            if staging.exists():
                for path in staging.rglob("*"):
                    if not path.is_symlink():
                        path.chmod(0o755 if path.is_dir() else 0o644)
                staging.chmod(0o755)
                shutil.rmtree(staging)


def publish_result(output, destination):
    result = {
        "cache_key": destination.name,
        "cache_path": str(destination),
        "model_manifest_sha256": destination.name,
        **json.loads((destination / "model-content.json").read_text()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
