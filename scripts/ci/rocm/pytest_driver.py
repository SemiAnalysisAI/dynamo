#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One isolated pytest process with installed-package provenance gates."""

import json
import os
import sys
from importlib import metadata
from pathlib import Path

import pytest

# The trusted controller is mounted outside Python site-packages.
sys.path.insert(0, "/results/controller")
from barite_verify import provenance


def main():
    suite, directory = sys.argv[1:]
    result = Path(directory)
    contract = json.loads(Path(__file__).with_name("contract.json").read_text())
    request = json.loads((result / "request.json").read_text())
    build = json.loads(Path("/opt/dynamo/ci-manifest.json").read_text())
    plugins = sorted(
        [e.name, e.value, e.dist.name, e.dist.version]
        for e in metadata.entry_points(group="pytest11")
    )
    assert plugins == build["pytest_plugins"], "Pytest plugin inventory changed"
    os.chdir("/workspace")
    # Required for repository tests.utils imports, not for Dynamo imports.
    sys.path.insert(0, "/workspace")

    class Evidence:
        def pytest_sessionstart(self, session):
            provenance(request, build)

        def pytest_collection_finish(self, session):
            actual = [item.nodeid for item in session.items]
            expected = contract["suites"][suite]
            if actual != expected:
                raise pytest.UsageError(
                    f"Frozen collection differs: {actual!r} != {expected!r}"
                )
            active = sorted(
                str(name)
                for name, plugin in session.config.pluginmanager.list_name_plugin()
                if plugin is not None
            )
            (result / (suite + "-collection.json")).write_text(
                json.dumps({"nodeids": actual, "plugins": active}, indent=2) + "\n"
            )
            provenance(request, build)

    options = [
        "--rootdir=/workspace",
        "-c",
        "/workspace/pyproject.toml",
        "-n",
        "0",
        "-xvv",
        "--import-mode=importlib",
        "--junitxml=" + str(result / "test-results" / (suite + ".xml")),
    ]
    if suite != "imports":
        options += ["--models-dir=/models"]
    if suite == "frontend":
        options += ["--timeout=600"]
    raise SystemExit(
        int(pytest.main(options + contract["suites"][suite], plugins=[Evidence()]))
    )


if __name__ == "__main__":
    main()
