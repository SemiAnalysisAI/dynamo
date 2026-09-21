#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download in a child whose writable HF/Xet cache environment is already set."""

import sys

from huggingface_hub import snapshot_download


def main():
    repo_id, revision, cache_dir = sys.argv[1:]
    snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
