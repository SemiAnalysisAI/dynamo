# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the official image-build sequence without starting a daemon."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm import build_images


class ImageBuildTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "container").mkdir(parents=True)
        (self.source / "container/rendered.Dockerfile").write_text("FROM pinned\n")
        (self.source / "container/Dockerfile.test").write_text(
            "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n"
        )
        self.results = self.root / "results"
        self.results.mkdir()
        (self.results / "request.json").write_text(
            json.dumps({"source_sha": "a" * 40, "run_key": "build-1"})
        )
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.daemon = MagicMock()
        self.daemon.__enter__.return_value = self.daemon
        self.daemon.command.side_effect = lambda *args: ["docker", *args]
        self.daemon.env = {}
        self.daemon.cgroup_parent = "/slurm/job_100"
        contract = build_images.read_json(
            Path(build_images.__file__).with_name("contract.json")
        )
        self.runtime = {
            "Id": "sha256:" + "b" * 64,
            "Config": {
                "Labels": {
                    "org.opencontainers.image.base.name": contract[
                        "base_uri"
                    ].removeprefix("docker://registry-1.docker.io#")
                }
            },
        }
        self.test = {"Id": "sha256:" + "c" * 64}
        self.environment = patch.dict(
            os.environ, {"SLURM_JOB_ID": "100", "SLURM_CPUS_PER_TASK": "16"}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def exercise(self, runner):
        with (
            patch.object(build_images, "DockerDaemon", return_value=self.daemon),
            patch.object(
                build_images, "inspect_image", side_effect=[self.runtime, self.test]
            ),
            patch.object(build_images, "run", side_effect=runner),
        ):
            build_images.build_images(self.source, self.results, self.scratch)

    def test_runtime_then_standard_test_image_then_exact_image_export(self):
        calls = []
        self.exercise(
            lambda command, log, **kwargs: calls.append((command, log, kwargs))
        )
        runtime = next(
            command for command, log, _ in calls if log.name == "build-runtime.log"
        )
        test = next(
            command for command, log, _ in calls if log.name == "build-test.log"
        )
        self.assertIn(str(self.source / "container/rendered.Dockerfile"), runtime)
        self.assertIn("DYNAMO_COMMIT_SHA=" + "a" * 40, runtime)
        self.assertIn(str(self.source / "container/Dockerfile.test"), test)
        self.assertIn("BASE_IMAGE=dynamo-rocm-runtime:build-1", test)
        self.assertIn("--target=test_image", test)
        logs = [log.name for _, log, _ in calls]
        self.assertLess(
            logs.index("runtime-sanity_check.log"), logs.index("build-test.log")
        )
        self.assertLess(
            logs.index("build-test.log"), logs.index("export-test-image.log")
        )
        sanity = next(
            command
            for command, log, _ in calls
            if log.name == "runtime-sanity_check.log"
        )
        self.assertIn(self.runtime["Id"], sanity)
        self.assertIn("/workspace/dev/sanity_check.py", sanity)
        inspect = next(
            command for command, log, _ in calls if log.name == "record-image.log"
        )
        self.assertIn(self.test["Id"], inspect)
        self.assertEqual(calls[-1][0][-1], "dockerd://dynamo-rocm-test:build-1")
        metadata = json.loads((self.results / "image-build.json").read_text())
        self.assertEqual(metadata["runtime_sanity"]["image_id"], self.runtime["Id"])
        self.assertEqual(metadata["test_image_id"], self.test["Id"])

    def test_failed_runtime_build_never_builds_or_exports_test_image(self):
        calls = []

        def fail_runtime(command, log, **kwargs):
            calls.append(log.name)
            if log.name == "build-runtime.log":
                raise subprocess.CalledProcessError(1, command)

        with self.assertRaises(subprocess.CalledProcessError):
            self.exercise(fail_runtime)
        self.assertNotIn("build-test.log", calls)
        self.assertNotIn("export-test-image.log", calls)
        self.assertFalse((self.results / "image-build.json").exists())
        self.daemon.__exit__.assert_called_once()

    def test_failed_runtime_sanity_blocks_test_image(self):
        calls = []

        def fail_sanity(command, log, **kwargs):
            calls.append(log.name)
            if log.name == "runtime-sanity_check.log":
                raise subprocess.CalledProcessError(1, command)

        with self.assertRaises(subprocess.CalledProcessError):
            self.exercise(fail_sanity)
        self.assertNotIn("build-test.log", calls)
        self.assertFalse((self.results / "image-build.json").exists())

    def test_wrong_base_image_blocks_qualification(self):
        self.runtime["Config"]["Labels"][
            "org.opencontainers.image.base.name"
        ] = "different-image"
        with self.assertRaisesRegex(ValueError, "different base image"):
            self.exercise(lambda *args, **kwargs: None)
        self.assertFalse((self.results / "image-build.json").exists())


if __name__ == "__main__":
    unittest.main()
