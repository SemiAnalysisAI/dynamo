#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
# Sanitize before the first Python interpreter, including user-controlled pip settings.
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT
unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTEST_DISABLE_PLUGIN_AUTOLOAD
unset PIP_TARGET PIP_PREFIX PIP_USER PIP_EXTRA_INDEX_URL PIP_INDEX_URL
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null
exec python3 -I "$(dirname "$0")/install_source.py" "$@"
