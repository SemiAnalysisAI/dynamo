<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Barite ROCm CI

This lane stages reviewed Dynamo source over SSH and builds/tests it in a one-GPU
Slurm allocation using Pyxis/Enroot. The login host only stages files and controls
Slurm. The implementation has offline tests; a successful HIP/Dynamo hardware run,
local artifact parity, and hosted Actions connectivity still require qualification.

## Prerequisites

- Use two clean checkouts: the reviewed controller commit containing these helpers,
  and the reviewed candidate commit. Commit identities are full lowercase Git SHAs.
  Materialize Git LFS files in the candidate (`git lfs pull`). Dirty trees, untracked
  files, unresolved LFS pointers, and submodules are rejected.
- Use Python 3.12 or newer, Git, and OpenSSH on the client. The remote account needs
  Python 3.12 or newer, Slurm commands, `stdbuf`, writable shared home storage, and
  access to Pyxis/Enroot on allocated compute nodes.
- Configure the local `barite-login` SSH alias and verify pinned host keys for every
  hop. CI uses dedicated, approved, noninteractive SSH keys; do not reuse a personal
  key or enable agent forwarding. `setup-ssh` expects `TW_JUMP_SSH_KEY`,
  `BARITE_SSH_KEY`, and `BARITE_KNOWN_HOSTS` in its environment.
- Local and CI SSH connections must reach the same approved account and artifact
  ownership domain. The default expected UID is `20011`; `--expected-uid` supports
  an explicitly approved alternative. The shared root is `$HOME/dynamo-rocm-ci`.
- Verify the site's GPU device-isolation configuration before concurrent shared-node
  qualification. A Slurm GPU mask alone does not establish device enforcement.
  Compute allocations need access to the pinned registry, package/dependency
  endpoints, and the frozen public model revision.

Before pushing/bootstrap on `SemiAnalysisAI/dynamo`, keep Actions disabled or
individually disable inherited workflows. Configure a protected `barite-rocm`
environment restricted to deployments from `main`, with required review, the two dedicated key secrets, and the pinned
`BARITE_KNOWN_HOSTS` variable. Bootstrap the reviewed workflow/controller on `main`,
then enable only `rocm-ci.yaml` after confirming inherited workflows remain disabled.
The new workflow is manual, fork-only, main-only, and serializes its own campaigns.
It does not serialize separately launched local jobs.

The fork bootstrap preserves inherited workflows in `.github/upstream-workflows/`,
outside GitHub's workflow discovery directory. This also prevents automatic runs
when inherited workflow IDs have not yet been indexed and cannot be disabled via
the API. Keep this fork-only change when syncing upstream.

## First local build

Run from the clean controller checkout. Replace the paths and SHAs below with
reviewed values; keep evidence outside both checkouts. Use a new run key for every
new allocation. The controller's helpers implement the build/test contract, while
the candidate archive supplies the Dynamo packages and selected repository tests.

```bash
CONTROLLER_SHA=$(git rev-parse HEAD)
CANDIDATE_DIR=/absolute/path/to/candidate
SOURCE_SHA=$(git -C "$CANDIDATE_DIR" rev-parse HEAD)
RUN_KEY=local-qualification-001
OUTPUT="$HOME/barite-results/$RUN_KEY"

./scripts/ci/barite-client.sh preflight --ssh-host barite-login --expected-uid 20011
./scripts/ci/barite-client.sh run \
  --ssh-host barite-login --expected-uid 20011 \
  --controller-sha "$CONTROLLER_SHA" \
  --source-dir "$CANDIDATE_DIR" --source-sha "$SOURCE_SHA" \
  --run-key "$RUN_KEY" --suite aggregate --partition auto --gpus 1 \
  --queue-timeout 30m --time-limit 04:00:00 --controller-timeout 285m \
  --output "$OUTPUT"
```

The default `--partition auto` policy prefers `compute-1` when current Slurm
evidence shows an eligible idle GPU node with enough CPU and memory. Otherwise it
falls back to an available `compute-0` partition. A failed or malformed scheduler
query stops submission; it never silently chooses a fallback. The selected
partition and query evidence are saved in `partition-selection.json`. An idle
snapshot is not a reservation: Slurm can still queue the job. Explicit
`--partition compute-1` and `--partition compute-0` are diagnostic overrides.

The fixed request uses one node/task/GPU, 16 CPUs, 64 GiB memory, and no automatic
requeue. The queue budget is 30 minutes; the allocation limit is four hours.
`sbatch --deadline=now+270minutes` bounds an abandoned queued request. The detached
remote monitor and client have 285-minute budgets; the Actions job has a
330-minute cap with reserved recovery/upload steps. Finalization accepts at most
240 seconds. These are qualification limits, not performance measurements.

The batch wrapper verifies/extracts source into unique node-local `/tmp` storage,
builds an installed-wheel overlay, saves and hashes the root filesystem, prepares
the pinned model cache, and runs tests in a fresh container step. It preserves the
scheduler's GPU assignment. Images are published as
`images/<sha256>/image.sqsh`; model snapshots live under `models/<manifest-hash>`.
No host Docker daemon or login-node compiler is used.

## Status, reconnect, and cleanup

Keep the original run key and evidence directory. `status` reports durable state;
`resume` reconciles the exact job and waits for its verdict without resubmission.
A lost SSH connection does not stop the bounded remote monitor.

```bash
./scripts/ci/barite-client.sh status \
  --ssh-host barite-login --run-key "$RUN_KEY" --output "$OUTPUT"
./scripts/ci/barite-client.sh resume \
  --ssh-host barite-login --run-key "$RUN_KEY" --output "$OUTPUT"
./scripts/ci/barite-client.sh finalize \
  --ssh-host barite-login --run-key "$RUN_KEY" --output "$OUTPUT" \
  --cancel-if-active --deadline-seconds 240
```

`finalize` reconciles/cancels only a job whose scheduler owner and run identity
match, and collects available evidence. It is not a replacement for a successful
`run`/`resume` verdict. Ambiguous recovery never submits another job or cancels a
set of jobs. Keep failed run directories for investigation; these commands do not
prune shared image/model caches.

Success requires the same job ID across a successful streamed `sbatch --wait`
receipt, explicit `scontrol` `COMPLETED`/`0:0` evidence with no restart, and a matching
atomic workload verdict. Missing scheduler records, a stale monitor heartbeat,
missing/incorrect manifests, missing-record warnings, failed steps, and incomplete
or skipped expected tests fail closed. An empty queue or a saved image is not a
passing result. Inspect `client-result.json`, `controller-result.json`,
`wait-result.json`, `terminal.json`, `completed.json`, and the collected logs.
Evidence collection is allowlisted and size-bounded; `collection-report.json`
records size omissions. Images, models, and SSH keys are not uploaded.

## Repeat the artifact, then qualify Actions

After a successful first local build, extract `sqsh_sha256` from
`$OUTPUT/image-manifest.json`. Run a second clean local allocation with the same
source/controller identities, a new run key/output directory, and
`--reuse-image-sha "$IMAGE_SHA"`. This skips rebuilding and verifies the exact
published image before testing. Do not claim byte parity for a newly rebuilt image.

After both local jobs pass, dispatch the protected workflow using those identities:

```bash
gh workflow run rocm-ci.yaml --repo SemiAnalysisAI/dynamo --ref main \
  -f controller_sha="$CONTROLLER_SHA" -f source_sha="$SOURCE_SHA" \
  -f reuse_image_sha="$IMAGE_SHA"
```

The controller SHA must equal the workflow's exact `main` commit. The candidate
must be reachable from the fork's `rocm-ci` branch. Actions uses sibling
`controller`/`candidate` checkouts and the same CLI/contract. Its artifact upload
runs after failure, including early checkout failures. Only after reuse parity is
qualified should a separate campaign omit `reuse_image_sha` to test a fresh
Actions build.

The contract binds source SHA and materialized archive hash, controller SHA and
bundle hash, base image digest, final SQSH hash, frozen model manifest, and the
expected test selection. Review `rocm/contract.json` when changing that contract.
These receipts establish reproducibility for reviewed code; they are not a sandbox
against malicious candidate code.

## Offline validation

No Slurm allocation or SSH connection is needed:

```bash
python3 -m unittest discover -s scripts/ci/tests -p 'test_*.py' -v
bash -n scripts/ci/barite-client.sh scripts/ci/barite.sbatch
bash -n scripts/ci/rocm/install-source.sh scripts/ci/rocm/run-tests.sh
```
