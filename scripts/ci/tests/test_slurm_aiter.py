# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise AITER cache seeding without importing AITER or accessing a GPU."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm import prepare_aiter_cache


class AiterCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "base/aiter/jit"
        self.source.mkdir(parents=True)
        self.module = self.source / "module.so"
        self.module.write_bytes(b"pinned prebuilt")
        self.destination = self.root / "job/cache"
        self.helper = prepare_aiter_cache

    def test_seed_is_idempotent_and_rebuild_does_not_modify_source(self):
        records = self.helper.seed_prebuilt_modules(self.source, self.destination)
        self.assertEqual(
            records, self.helper.seed_prebuilt_modules(self.source, self.destination)
        )
        link = self.destination / "module.so"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), self.module)
        self.assertEqual(
            records,
            [
                {
                    "name": "module.so",
                    "target": str(self.module),
                    "size": len(b"pinned prebuilt"),
                }
            ],
        )
        # AITER 0.1.19 removes its output before installing a rebuilt extension.
        link.unlink()
        link.write_bytes(b"rebuilt for this job")
        self.assertEqual(self.module.read_bytes(), b"pinned prebuilt")
        self.assertEqual(link.read_bytes(), b"rebuilt for this job")
        with self.assertRaisesRegex(ValueError, "without source provenance"):
            self.helper.seed_prebuilt_modules(self.source, self.destination)
        self.assertEqual(link.read_bytes(), b"rebuilt for this job")

    def test_escaping_source_symlink_is_rejected(self):
        outside = self.root / "outside.so"
        outside.write_bytes(b"untrusted")
        (self.source / "escape.so").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes installed"):
            self.helper.seed_prebuilt_modules(self.source, self.destination)
        self.assertFalse((self.destination / "escape.so").exists())

    def test_conflicting_destination_and_empty_source_are_rejected(self):
        for destination in (self.source, self.root / "alias"):
            if destination != self.source:
                destination.symlink_to(self.root / "base", target_is_directory=True)
            with (
                self.subTest(destination=destination),
                self.assertRaisesRegex(ValueError, "separate writable"),
            ):
                self.helper.seed_prebuilt_modules(self.source, destination)
        self.destination.mkdir(parents=True)
        wrong = self.root / "wrong.so"
        wrong.write_bytes(b"wrong")
        (self.destination / "module.so").symlink_to(wrong)
        with self.assertRaisesRegex(ValueError, "Unexpected AITER cache symlink"):
            self.helper.seed_prebuilt_modules(self.source, self.destination)
        self.module.unlink()
        with self.assertRaisesRegex(ValueError, "no prebuilt"):
            self.helper.seed_prebuilt_modules(self.source, self.destination)

    def test_metadata_gate_and_evidence(self):
        output = self.root / "evidence.json"
        distribution = SimpleNamespace(
            version="0.1.19", locate_file=lambda _: self.source
        )
        with (
            patch.object(
                self.helper.sys,
                "argv",
                [
                    "prepare",
                    "--directory",
                    str(self.destination),
                    "--output",
                    str(output),
                ],
            ),
            patch.object(self.helper.sys, "base_prefix", str(self.root / "base")),
            patch.object(
                self.helper.metadata, "distribution", return_value=distribution
            ),
            patch("builtins.print"),
        ):
            self.helper.main()
            evidence = json.loads(output.read_text())
            self.assertEqual(evidence["source"], str(self.source))
            self.assertEqual(evidence["version"], "0.1.19")
            self.assertEqual(len(evidence["modules"]), 1)
            distribution.version = "0.1.20"
            with self.assertRaisesRegex(ValueError, "Expected amd-aiter"):
                self.helper.main()
            distribution.version = "0.1.19"
            with (
                patch.object(self.helper.sys, "base_prefix", str(self.root / "other")),
                self.assertRaisesRegex(ValueError, "installed base interpreter"),
            ):
                self.helper.main()


if __name__ == "__main__":
    unittest.main()
