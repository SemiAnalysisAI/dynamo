#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT
unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTEST_DISABLE_PLUGIN_AUTOLOAD
export PYTHONNOUSERSITE=1 VIRTUAL_ENV=/opt/dynamo/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
# Libraries can resolve/cache these paths at import time. Configure them before
# the first Python process; the freshly loaded rootfs and /models are read-only.
cache_root=/workspace/.ci-test-cache
mkdir -p "$cache_root/tmp"
export XDG_CACHE_HOME="$cache_root"
export XDG_CONFIG_HOME="$cache_root/config"
export VLLM_CONFIG_ROOT="$cache_root/config/vllm"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TRITON_CACHE_DIR="$cache_root/triton"
export CUPY_CACHE_DIR="$cache_root/cupy"
export TORCH_EXTENSIONS_DIR="$cache_root/torch-extensions"
export VLLM_CACHE_ROOT="$cache_root/vllm"
export TMPDIR="$cache_root/tmp"
export HF_HOME=/models HF_HUB_CACHE=/models/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AITER_JIT_DIR="$cache_root/aiter"
/opt/dynamo/venv/bin/python3 -I "$(dirname "$0")/prepare_aiter_cache.py" \
    --directory "$AITER_JIT_DIR" --output /results/aiter-cache.json
exec /opt/dynamo/venv/bin/python3 -I "$(dirname "$0")/run_tests.py" "$@"
