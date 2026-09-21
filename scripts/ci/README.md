<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ROCm CI on Slurm

This optional lane builds Dynamo's rendered ROCm runtime Docker image, layers the
standard test image on top, and runs one-GPU smoke tests through Pyxis/Enroot.
SSH connects to a login host for staging and scheduler control; builds and tests
run in a Slurm allocation on compute nodes. The same client drives local
qualification and the manual GitHub Actions workflow.

Pull requests changing this lane or the container sources run dependency-free
controller tests, Dockerfile rendering tests, and shell syntax checks on a
GitHub-hosted runner. Rendering tests install Jinja2 and PyYAML in a separate
environment. These jobs do not receive cluster credentials. GPU jobs require an
explicit workflow dispatch and the protected environment, after offline checks
pass.

## Requirements and configuration

- Client: Python 3.12 or newer, Git with Git LFS, and OpenSSH.
- Cluster: Python 3.12 or newer, Slurm commands, `stdbuf`, shared home storage,
  Pyxis/Enroot, and an MI300X-compatible ROCm environment. Compute nodes also need
  Docker client/daemon binaries, containerd 2.x, Enroot's `dockerd://` import support, and
  noninteractive `sudo` for a private rootful Docker daemon. They need access to
  the pinned registry, dependency endpoints, and public model revision.
- Verify Slurm device enforcement before sharing nodes. The lane checks cgroup
  device isolation and GPU resource configuration; a GPU visibility mask alone
  does not provide device isolation.
- Use clean controller and candidate checkouts with full lowercase commit SHAs.
  Materialize candidate LFS files with `git lfs pull`. Dirty trees, untracked files,
  unresolved LFS pointers, and submodules are rejected.
- Local and Actions runs must use the same remote account and shared artifact
  root, `$HOME/dynamo-rocm-ci`. Pass that account's numeric UID explicitly with
  `--expected-uid` on every remote command.

For Actions, create a protected environment named `rocm-slurm`, require review,
and restrict it to the repository's default branch. Configure:

| Setting | Location | Purpose |
| --- | --- | --- |
| `ROCM_SLURM_ENABLED` | Repository variable | Set to `true` to enable this manual lane. |
| `ROCM_SLURM_SOURCE_BRANCH` | Environment variable | Trusted branch containing reviewed candidate commits. |
| `SLURM_UID` | Environment variable | Numeric UID of the remote account. |
| `SLURM_SSH_CONFIG` | Environment variable | JSON connection route, described below. |
| `SLURM_KNOWN_HOSTS` | Environment variable | Independently verified host keys for every SSH hop. |
| `SLURM_SSH_KEY` | Environment secret | Dedicated, unencrypted cluster private key. |
| `SLURM_GATEWAY_SSH_KEY` | Environment secret, optional | Dedicated private key for hops using the `gateway` identity. |
| `SLURM_PREFERRED_PARTITION` | Environment variable, optional | Preferred partition; defaults to `compute-1`. |
| `SLURM_FALLBACK_PARTITION` | Environment variable, optional | Fallback partition; defaults to `compute-0`. |

The connection route accepts a login host and zero to four ordered jump hosts.
Each host requires `hostname` and `user`; `port` defaults to `22` and `identity`
defaults to `cluster`. The two identities select the corresponding secrets:

```json
{
  "login": {"hostname": "login.example", "user": "ci", "identity": "cluster"},
  "jumps": [
    {"hostname": "gateway.example", "user": "ci", "port": 22, "identity": "gateway"}
  ]
}
```

`setup-ssh --directory PATH` reads these SSH settings from the process environment
and writes a private configuration with the alias `ci-slurm-login`. It enables
strict host-key checks and disables agent forwarding. Local runs can instead use
an existing verified SSH alias. Keep private keys out of the repository.

## Local qualification

Run from the clean controller checkout. Set the remote UID and SSH alias to your
configured account, use a fresh run key, and keep evidence outside both checkouts:

```bash
CONTROLLER_SHA=$(git rev-parse HEAD)
CANDIDATE_DIR=/absolute/path/to/candidate
SOURCE_SHA=$(git -C "$CANDIDATE_DIR" rev-parse HEAD)
SSH_HOST=slurm-login
SLURM_UID=12345  # Replace with the remote account UID.
RUN_KEY=local-qualification-001
OUTPUT="$HOME/slurm-results/$RUN_KEY"

./scripts/ci/slurm-client.sh preflight \
  --ssh-host "$SSH_HOST" --expected-uid "$SLURM_UID"
./scripts/ci/slurm-client.sh run \
  --ssh-host "$SSH_HOST" --expected-uid "$SLURM_UID" \
  --controller-sha "$CONTROLLER_SHA" \
  --source-dir "$CANDIDATE_DIR" --source-sha "$SOURCE_SHA" \
  --run-key "$RUN_KEY" --suite aggregate --partition auto \
  --preferred-partition compute-1 --fallback-partition compute-0 \
  --gpus 1 --queue-timeout 30m --time-limit 04:00:00 --controller-timeout 285m \
  --output "$OUTPUT"
```

Add `--ssh-config PATH` when using a configuration outside `~/.ssh/config`.
With `--partition auto`, the client prefers an eligible idle GPU node in the
preferred partition and otherwise selects the available fallback partition.
This implements the `compute-1` then `compute-0` policy with the defaults above.
Failed or malformed scheduler queries stop submission. The chosen partition and
query evidence are recorded in `partition-selection.json`. An idle snapshot is
not a reservation, so Slurm can still queue the job.

Each request uses one node, one task, one GPU, 16 CPUs, 64 GiB memory, and no
automatic requeue. Queue time is limited to 30 minutes and allocation time to four
hours. A Slurm deadline also bounds abandoned queued jobs. The remote monitor and
client are limited to 285 minutes; Actions allows 330 minutes for recovery and
artifact upload. The workflow serializes Actions campaigns; local runs are
scheduled independently.

The batch job extracts verified source into node-local scratch and builds two
images using Docker's BuildKit engine and a pinned Buildx plugin:

1. Render `container/render.py --framework vllm --device rocm --target runtime`
   for `linux/amd64`, then build the runtime Dockerfile.
2. Run `dev/sanity_check.py --runtime-check --no-gpu-check` and `pip check` in
   that runtime image before adding test dependencies.
3. Build `container/Dockerfile.test`'s `test_image` target with the runtime image
   as `BASE_IMAGE`.
4. Import the resulting test image directly from the private Docker daemon with
   Enroot, hash the SQSH, and publish it for Pyxis execution.

Docker and containerd use private sockets and storage for each allocation. The cgroup parent
keeps builds inside the Slurm resource allocation; cleanup stops the private
daemon. The lane uses no shared host Docker socket and runs no compiler on the
login host. Runtime and test builds are ordinary repository Docker builds;
test execution installs no additional packages.
Image inspection uses node-local bind mounts so root-squashed shared homes
work with the private daemon. The submitting account copies the evidence back
to shared storage, including partial diagnostics when inspection fails.

`image-build.json` records both Dockerfile hashes, both local OCI image config
digests, and the runtime sanity results. The collected Dockerfiles and
`runtime-image-inspect.json` / `test-image-inspect.json` preserve the build inputs
and image metadata. `image-manifest.json` binds this evidence and installed wheel
provenance to the source, controller, and final SQSH identities.

The test job prepares the pinned model snapshot and starts a fresh container.
Images and model snapshots remain read-only during tests. Runtime caches use
private scratch, including AITER's JIT cache and Dynamo's native
`$HOME/.cache/dynamo/mdc` directory; the host home is not mounted.

The fixed test contract in [`rocm/contract.json`](rocm/contract.json) includes a HIP
smoke check, six lazy-import cases, one mocker frontend test, and one aggregated
vLLM serving test. Services use loopback and NATS events. Listener evidence covers
the five named service ports and TCP listeners on the frontend and worker system
endpoint PIDs. It does not inventory unrelated engine subprocesses.

## Status and recovery

Reuse the original run key and output directory. `resume` waits for the existing
job without resubmitting; the bounded remote monitor survives an SSH disconnect.

```bash
./scripts/ci/slurm-client.sh status \
  --ssh-host "$SSH_HOST" --expected-uid "$SLURM_UID" \
  --run-key "$RUN_KEY" --output "$OUTPUT"
./scripts/ci/slurm-client.sh resume \
  --ssh-host "$SSH_HOST" --expected-uid "$SLURM_UID" \
  --run-key "$RUN_KEY" --output "$OUTPUT"
./scripts/ci/slurm-client.sh finalize \
  --ssh-host "$SSH_HOST" --expected-uid "$SLURM_UID" \
  --run-key "$RUN_KEY" --output "$OUTPUT" \
  --cancel-if-active --deadline-seconds 240
```

`finalize` verifies scheduler ownership and run identity before cancelling an
active job, and collects partial evidence. It does not establish a passing
verdict. Ambiguous recovery neither resubmits nor cancels unrelated jobs. Keep
failed run directories for investigation; these commands do not prune shared
image or model caches. Remove generated SSH files with
`cleanup-ssh --directory PATH` when finished.

## Reuse the image, then qualify Actions

A pass requires matching job IDs across the successful `sbatch --wait` receipt,
Slurm `COMPLETED`/`0:0` evidence with zero restarts, and the workload verdict. All
selected tests must pass without skips. Missing scheduler records, stale monitor
heartbeats, and mismatched or incomplete manifests fail validation. Inspect
`client-result.json`, `controller-result.json`, `wait-result.json`,
`terminal.json`, `completed.json`, and the logs.

After the first local pass, read `sqsh_sha256` from `image-manifest.json`. Repeat
the local command with identical source and controller SHAs, a fresh run key and
output directory, and `--reuse-image-sha "$IMAGE_SHA"`. This verifies the exact
published artifact in a new allocation. A rebuilt image does not establish byte
parity.

After both local runs pass, dispatch Actions from the same repository:

```bash
REPOSITORY=OWNER/REPOSITORY  # Set the repository configured for this cluster.
DEFAULT_BRANCH=$(gh repo view "$REPOSITORY" --json defaultBranchRef --jq .defaultBranchRef.name)
gh workflow run rocm-ci.yaml --repo "$REPOSITORY" --ref "$DEFAULT_BRANCH" \
  -f controller_sha="$CONTROLLER_SHA" -f source_sha="$SOURCE_SHA" \
  -f reuse_image_sha="$IMAGE_SHA"
```

The controller SHA must equal the workflow's default-branch commit. The candidate
must be reachable from `ROCM_SLURM_SOURCE_BRANCH`. Actions uses separate controller
and candidate checkouts, the same CLI, and the same artifact contract. It attempts
finalization and evidence upload even after failure. Collection is allowlisted
and size-bounded; `collection-report.json` records omissions. Images, model
weights, and SSH keys are excluded from uploads.

Source or controller changes require a new qualification campaign. These
provenance checks establish reproducibility for reviewed code; they do not
sandbox malicious candidate code. After image reuse is qualified, a separate
Actions run without `reuse_image_sha` can validate a fresh remote build.

## Offline checks

```bash
python3 -m unittest discover -s scripts/ci/tests -p 'test_*.py' -v
for script in scripts/ci/slurm-client.sh scripts/ci/slurm.sbatch \
  scripts/ci/rocm/run-tests.sh; do
  bash -n "$script"
done
```

Dockerfile rendering tests additionally require Jinja2 and PyYAML in your Python
environment:

```bash
python3 -m pip install Jinja2==3.1.6 PyYAML==6.0.3
python3 -m unittest discover -s container/tests -p 'test_*.py' -v
```
