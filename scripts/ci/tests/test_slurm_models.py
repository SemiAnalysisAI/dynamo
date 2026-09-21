# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline model publication checks; never contacts Hugging Face or Slurm."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm import prepare_models


class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.cleanup)
        self.models = prepare_models
        self.cache = self.root / "models"
        self.output = self.root / "model-manifest.json"
        self.argv = [
            prepare_models.__file__,
            "--cache-root",
            str(self.cache),
            "--output",
            str(self.output),
        ]

    def cleanup(self):
        # Published directories deliberately become read-only.
        for directory, _, files in os.walk(self.root):
            Path(directory).chmod(0o755)
            for name in files:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(0o644)
        self.temporary.cleanup()

    def download(self, command, *, env, check):
        self.assertTrue(check)
        self.assertEqual(command[1], "-I")
        self.assertEqual(Path(command[2]).name, "download_model.py")
        repo_id, revision, destination = command[3:]
        hub = Path(destination)
        scratch = Path(env["TMPDIR"])
        self.assertEqual(env["HF_HUB_CACHE"], str(hub))
        self.assertTrue(scratch.name.startswith(".download-"))
        self.assertTrue(hub.parent.name.startswith(".preparing-"))
        self.assertFalse(scratch.is_relative_to(hub.parent))
        for key in ("HF_HOME", "HF_XET_CACHE", "HF_ASSETS_CACHE", "XDG_CACHE_HOME"):
            path = Path(env[key])
            self.assertTrue(path.is_relative_to(scratch), key)
            path.mkdir(parents=True, exist_ok=True)
            (path / "transient").write_text("not model content")
        repo = hub / ("models--" + repo_id.replace("/", "--"))
        (repo / "blobs").mkdir(parents=True)
        (repo / "blobs/weights").write_bytes(b"test weights")
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").symlink_to("../../blobs/weights")
        (snapshot / "config.json").write_text("{}")
        (hub / ".locks").mkdir()
        (hub / ".locks/download.lock").touch()

    def test_cache_environment_precedes_child_and_transients_are_excluded(self):
        with (
            patch.object(self.models.sys, "argv", self.argv),
            patch.dict(
                os.environ, {"HF_HOME": "/read-only", "HF_XET_CACHE": "/read-only/xet"}
            ),
            patch.object(
                self.models.subprocess, "run", side_effect=self.download
            ) as download,
        ):
            self.models.main()
            download.assert_called_once()
            manifest = json.loads(self.output.read_text())
            published = Path(manifest["cache_path"])
            self.assertEqual(
                published.name, self.models.digest(published / "model-content.json")
            )
            self.assertTrue(manifest["files"])
            self.assertTrue(
                all(name.startswith("hub/models--") for name in manifest["files"])
            )
            self.assertFalse(
                any(
                    "transient" in name or ".locks" in name
                    for name in manifest["files"]
                )
            )
            self.assertEqual(
                (published / "hub/models--Qwen--Qwen3-0.6B/refs/main").read_text(),
                manifest["model"]["revision"],
            )
            self.assertFalse(list(self.cache.glob(".download-*")))
            self.assertFalse(list(self.cache.glob(".preparing-*")))
            self.models.main()
            download.assert_called_once()  # Verified cache reuse must not redownload.
            self.assertEqual(json.loads(self.output.read_text()), manifest)
            weights = published / "hub/models--Qwen--Qwen3-0.6B/blobs/weights"
            weights.chmod(0o644)
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "content verification"):
                self.models.main()
            download.assert_called_once()

    def test_failed_child_cleans_both_staging_and_download_scratch(self):
        def fail(command, *, env, check):
            self.download(command, env=env, check=check)
            raise subprocess.CalledProcessError(1, command)

        with (
            patch.object(self.models.sys, "argv", self.argv),
            patch.object(self.models.subprocess, "run", side_effect=fail),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            self.models.main()
        self.assertFalse(self.output.exists())
        self.assertEqual(
            [path.name for path in self.cache.iterdir()], [".publish.lock"]
        )


if __name__ == "__main__":
    unittest.main()
