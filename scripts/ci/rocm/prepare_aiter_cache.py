#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the pinned image's AITER prebuilts in a writable per-job JIT directory."""

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

EXPECTED_VERSION = "0.1.19"


def seed_prebuilt_modules(source: Path, destination: Path) -> list[dict]:
    source = source.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or source == destination.resolve():
        raise ValueError("AITER JIT destination must be a separate writable directory")
    modules = sorted(source.glob("*.so"))
    if not modules:
        raise ValueError("Pinned AITER image has no prebuilt JIT modules")
    records = []
    for module in modules:
        target = module.resolve(strict=True)
        if not target.is_file() or not target.is_relative_to(source):
            raise ValueError(
                f"AITER prebuilt target escapes installed JIT directory: {module}"
            )
        link = destination / module.name
        if link.is_symlink():
            if link.resolve(strict=True) != target:
                raise ValueError(f"Unexpected AITER cache symlink: {link}")
        elif link.exists():
            raise ValueError(
                f"AITER cache entry already exists without source provenance: {link}"
            )
        else:
            link.symlink_to(target)
        records.append(
            {"name": module.name, "target": str(target), "size": target.stat().st_size}
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Metadata lookup does not import aiter: importing it initializes JIT state.
    distribution = metadata.distribution("amd-aiter")
    if distribution.version != EXPECTED_VERSION:
        raise ValueError(
            f"Expected amd-aiter {EXPECTED_VERSION}, got {distribution.version}"
        )
    source = Path(distribution.locate_file("aiter/jit")).resolve(strict=True)
    if not source.is_relative_to(Path(sys.base_prefix).resolve()):
        raise ValueError(
            "AITER prebuilts must come from the installed base interpreter"
        )
    modules = seed_prebuilt_modules(source, args.directory)
    # AITER 0.1.19 unlinks an existing target before copying a rebuilt module.
    # Thus a genuine required rebuild replaces our link, never the base .so.
    evidence = {
        "version": distribution.version,
        "source": str(source),
        "directory": str(args.directory),
        "modules": modules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Preserved {len(modules)} AITER {distribution.version} prebuilt modules")


if __name__ == "__main__":
    main()
