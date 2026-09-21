#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT
unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTEST_DISABLE_PLUGIN_AUTOLOAD
export PYTHONNOUSERSITE=1 VIRTUAL_ENV=/opt/dynamo/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
exec /opt/dynamo/venv/bin/python3 -I "$(dirname "$0")/run_tests.py" "$@"
