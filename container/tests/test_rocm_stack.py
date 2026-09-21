# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard the inherited ROCm packages against replacement by dependency installs."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "deps/vllm/rocm_stack.py"
SPEC = importlib.util.spec_from_file_location("rocm_stack", SCRIPT)
rocm_stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rocm_stack)


class RocmStackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.modules = {}
        for name in ("torch", "vllm"):
            module_file = self.root / "system" / name / "__init__.py"
            module_file.parent.mkdir(parents=True)
            module_file.write_text(f"# {name}\n")
            self.modules[name] = SimpleNamespace(__file__=str(module_file))
        self.modules["torch"].version = SimpleNamespace(hip="7.2.3", cuda=None)
        self.versions = {"torch": "2.11.0+rocm7.2.3", "vllm": "0.29.0"}
        patches = (
            mock.patch.object(rocm_stack, "PREFIX", self.root / "dynamo"),
            mock.patch.object(rocm_stack, "import_module", self.modules.__getitem__),
            mock.patch.object(
                rocm_stack.metadata, "version", self.versions.__getitem__
            ),
            mock.patch.object(
                rocm_stack.metadata,
                "distributions",
                return_value=[
                    SimpleNamespace(name=name, version=version)
                    for name, version in {
                        **self.versions,
                        "pytorch_triton_rocm": "3.5.0",
                        "ROCm-SDK": "7.2.3",
                        "requests": "2.32.5",
                    }.items()
                ],
            ),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def run_action(self, action):
        with mock.patch("sys.argv", [str(SCRIPT), action]):
            rocm_stack.main()

    def test_original_packages_pass_with_constraints_for_accelerator_dependencies(self):
        self.run_action("record")
        self.run_action("verify")
        constraints = (self.root / "dynamo/rocm-constraints.txt").read_text()
        for requirement in (
            "torch==2.11.0+rocm7.2.3",
            "vllm==0.29.0",
            "pytorch_triton_rocm==3.5.0",
            "ROCm-SDK==7.2.3",
        ):
            self.assertIn(requirement, constraints.splitlines())
        self.assertNotIn("requests", constraints)

    def test_same_version_package_shadowed_in_venv_fails(self):
        self.run_action("record")
        replacement = self.root / "dynamo/venv/torch/__init__.py"
        replacement.parent.mkdir(parents=True)
        replacement.write_text(Path(self.modules["torch"].__file__).read_text())
        self.modules["torch"].__file__ = str(replacement)
        with self.assertRaisesRegex(ValueError, "replaced or modified"):
            self.run_action("verify")

    def test_modified_base_package_fails(self):
        self.run_action("record")
        Path(self.modules["vllm"].__file__).write_text("# replacement\n")
        with self.assertRaisesRegex(ValueError, "replaced or modified"):
            self.run_action("verify")

    def test_non_rocm_torch_fails_before_recording(self):
        self.modules["torch"].version = SimpleNamespace(hip=None, cuda="13.0")
        with self.assertRaisesRegex(ValueError, "ROCm PyTorch"):
            self.run_action("record")
        self.assertFalse((self.root / "dynamo/rocm-base.json").exists())


if __name__ == "__main__":
    unittest.main()
