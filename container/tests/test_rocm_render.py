# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROCm renderer contract tests; run with unittest and Jinja2/PyYAML installed."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

CONTAINER = Path(__file__).resolve().parents[1]


class RocmRenderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.container = Path(self.directory.name)
        for name in ("render.py", "context.yaml", "Dockerfile.template"):
            shutil.copy2(CONTAINER / name, self.container / name)
        shutil.copytree(CONTAINER / "templates", self.container / "templates")

    def render(self, *arguments):
        return subprocess.run(
            [
                sys.executable,
                str(self.container / "render.py"),
                "--device",
                "rocm",
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_runtime_uses_digest_pinned_base_and_native_wheel_stage(self):
        result = self.render("--output-short-filename")
        self.assertEqual(result.returncode, 0, result.stderr)
        dockerfile = (self.container / "rendered.Dockerfile").read_text()
        context = yaml.safe_load((self.container / "context.yaml").read_text())
        base = context["vllm"]["rocm"]["base_image"]
        self.assertRegex(base, r"^vllm/vllm-openai-rocm@sha256:[a-f0-9]{64}$")
        self.assertIn(f"ARG BASE_IMAGE={base}\n", dockerfile)
        self.assertIn(f"ARG RUNTIME_IMAGE={base}\n", dockerfile)
        self.assertIn(f'LABEL org.opencontainers.image.base.name="{base}"', dockerfile)
        stages = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        self.assertEqual(
            stages,
            [
                "FROM ${BASE_IMAGE} AS dynamo_base",
                "FROM dynamo_base AS wheel_builder",
                "FROM ${RUNTIME_IMAGE} AS runtime",
            ],
        )
        self.assertIn("maturin build --locked --release", dockerfile)
        self.assertIn("COPY --from=wheel_builder /opt/dynamo/dist/", dockerfile)
        self.assertNotIn("scripts/ci", dockerfile)
        self.assertNotIn("slurm", dockerfile.lower())

    def test_non_cuda_output_name(self):
        result = self.render()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(
            (self.container / "vllm-runtime-rocm-amd64-rendered.Dockerfile").is_file()
        )

    def test_unsupported_rocm_variants_fail_before_writing_a_dockerfile(self):
        for arguments in (
            ("--framework", "sglang"),
            ("--framework", "dynamo"),
            ("--target", "dev"),
            ("--target", "wheel_builder"),
            ("--platform", "linux/arm64"),
            ("--platform", "linux/amd64,linux/arm64"),
            ("--make-efa",),
        ):
            with self.subTest(arguments=arguments):
                result = self.render("--output-short-filename", *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ROCm currently supports only", result.stderr)
                self.assertFalse((self.container / "rendered.Dockerfile").exists())


if __name__ == "__main__":
    unittest.main()
