<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Inactive upstream workflows in the SemiAnalysisAI fork

This fork keeps inherited NVIDIA workflows outside GitHub's `.github/workflows/`
discovery directory. Their original contents are preserved here. The only active
repository workflow is the manual Barite ROCm lane, `../workflows/rocm-ci.yaml`.

This is a fork bootstrap change, not an upstream Dynamo change. Do not copy these
files back during an upstream sync unless their triggers, credentials, compute
resources, and costs have been reviewed. Disable repository Actions before
publishing any branch that restores inherited workflow files; verify the active
workflow inventory before enabling Actions again.

GitHub did not index inherited workflow IDs while repository Actions was disabled,
so the API could not disable them individually before bootstrap. Keeping the files
outside the discovery directory avoids an enable-then-disable race.
