<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ROCm CI on Slurm

The [ROCm Slurm CI workflow](../../.github/workflows/rocm-ci.yaml) builds Dynamo's
rendered ROCm runtime Docker image, layers the standard test image on top, and
runs one-GPU smoke tests through Pyxis/Enroot. GitHub Actions is the execution
entry point. Its hosted runner uses SSH to stage the source and control Slurm;
image builds and GPU tests run inside an allocation on compute nodes.

Pull requests changing this lane or the container sources run controller,
Dockerfile-rendering, and shell checks without cluster credentials. GPU execution
requires manual dispatch, repository opt-in, an allowed deployment branch, and
approval through the configured GitHub environment.

## Configure the workflow

The hosted runner needs Python 3.12 or newer, Git LFS, and OpenSSH. The workflow
pins its checkout to `github.sha` and materializes LFS files. The same commit
supplies both controller and source; callers cannot select a different source
SHA, SSH alias, local checkout, run key, or prebuilt image.

Compute nodes need Python 3.12 or newer, Slurm commands, `stdbuf`, Pyxis/Enroot,
Docker client/daemon binaries, containerd 2.x, and noninteractive `sudo` for a
private Docker daemon. Enroot must support `dockerd://` imports. The cluster needs
shared home storage and access to the pinned registry, dependencies, and public
model revision. Verify Slurm device enforcement before sharing nodes: a GPU
visibility mask alone does not provide isolation.

Create a GitHub environment named `rocm-slurm`, or select an existing environment
with `ROCM_SLURM_ENVIRONMENT`. Require deployment review and allow only the
reviewed branches that may execute this workflow. An exact review-branch policy
allows testing before merge while retaining the approval checkpoint.

| Setting | Location | Purpose |
| --- | --- | --- |
| `ROCM_SLURM_ENABLED` | Repository variable | Set to `true` to enable the manual GPU job. |
| `ROCM_SLURM_ENVIRONMENT` | Repository variable, optional | Protected environment name; defaults to `rocm-slurm`. |
| `SLURM_UID` | Environment variable | Numeric UID of the remote non-root account. |
| `SLURM_SSH_CONFIG` | Environment variable | OpenSSH host configuration, described below. |
| `SLURM_KNOWN_HOSTS` | Environment variable | Independently verified host keys for every SSH hop. |
| `SLURM_SSH_KEY` | Environment secret | Dedicated, unencrypted cluster private key. |
| `SLURM_GATEWAY_SSH_KEY` | Environment secret, optional | Dedicated private key for jump hosts. |
| `SLURM_PREFERRED_PARTITION` | Environment variable, optional | Preferred partition; defaults to `compute-1`. |
| `SLURM_FALLBACK_PARTITION` | Environment variable, optional | Fallback partition; defaults to `compute-0`. |

Set `SLURM_SSH_CONFIG` to standard OpenSSH host blocks defining the
`ci-slurm-login` alias and any jump hosts. The workflow sets `ROCM_SSH_DIR` to
its temporary credential directory; OpenSSH expands this variable in identity
paths. For example, a route with one jump host uses:

```sshconfig
Host ci-slurm-gateway
  HostName gateway.example
  User ci
  IdentityFile ${ROCM_SSH_DIR}/gateway_key

Host ci-slurm-login
  HostName login.example
  User ci
  IdentityFile ${ROCM_SSH_DIR}/ssh_key
  ProxyJump ci-slurm-gateway
```

The workflow creates private SSH files under `RUNNER_TEMP`, enables strict host
key checking, disables agent forwarding, and removes those files during cleanup.
Keep keys and site-specific connection details out of the repository.

## Run and inspect CI

Select **ROCm Slurm CI → Run workflow** in GitHub Actions, choose the reviewed
branch, and approve the environment deployment after reviewing the run's commit.
The equivalent GitHub CLI invocation is:

```bash
gh workflow run rocm-ci.yaml --repo OWNER/REPOSITORY --ref REVIEWED_BRANCH
```

The workflow must already exist on the repository's default branch for GitHub to
offer manual dispatch. Its definition and source are both taken from the selected
branch's immutable run commit. The driver also verifies `GITHUB_WORKFLOW_SHA`,
`GITHUB_SHA`, and the checked-out commit agree.

Every run builds fresh images. Rerun a failed job through GitHub Actions after
inspecting its logs and artifacts; a new attempt receives a distinct run key.
There is no standalone SSH run/resume interface or image-reuse mode. Runs using
the same protected environment share a concurrency group.

## Build and test contract

The scheduler prefers an eligible idle GPU node in `compute-1`, then falls back
to `compute-0`. The variables above can select other partitions. Invalid scheduler
queries stop submission. `partition-selection.json` records the selection and
query evidence; an idle snapshot is not a reservation.

Each request uses one node, one task, one GPU, 16 CPUs, 64 GiB memory, and no
automatic requeue. Queue time is limited to 30 minutes and allocation time to four
hours. A Slurm deadline bounds abandoned queued jobs. The controller is limited
to 285 minutes; Actions allows 330 minutes for recovery and artifact upload.

Within the allocation, the workflow's controller:

1. Extracts the verified source into node-local scratch and renders
   `container/render.py --framework vllm --device rocm --target runtime` for
   `linux/amd64`, then builds the runtime Dockerfile with BuildKit.
2. Runs `dev/sanity_check.py --runtime-check --no-gpu-check` and `pip check` in
   that runtime image before adding test dependencies.
3. Builds `container/Dockerfile.test`'s `test_image` target with the runtime image
   as `BASE_IMAGE`.
4. Checks installed-wheel provenance, imports the exact test image from the
   private Docker daemon into Enroot, hashes the SQSH, and starts fresh Pyxis
   containers for model preparation and testing.

Docker and containerd use private sockets and storage. Build processes stay in
the Slurm allocation's cgroup. Image inspection uses node-local mounts and an
explicit home/cache for the Slurm UID, then copies evidence to shared storage.
Cleanup stops the private daemons. Test execution installs no extra packages.

The fixed test contract in [`rocm/contract.json`](rocm/contract.json) includes HIP
arithmetic on one MI300X, six lazy-import cases, one mocker frontend test, and one
aggregate vLLM serving test using the pinned model revision. Images and model
snapshots are read-only during testing. Runtime caches use private scratch.
Services use loopback and NATS events; listener evidence covers the named service
ports and the frontend/worker system endpoint PIDs.

This lane qualifies single-node aggregate smoke coverage. It does not qualify
native NIXL, KVBM, GPU memory service, multi-node transport, or performance.

## Evidence and failure recovery

The workflow always attempts bounded finalization, artifact upload, and SSH
cleanup. Finalization verifies run ownership before cancelling an active job and
preserves partial evidence. Ambiguous recovery does not resubmit or cancel an
unrelated job. A remote monitor survives transient runner SSH disconnections.

A passing result requires all selected tests without skips, matching source and
controller identities, installed-wheel origins, image/model hashes, loopback
listeners, and matching job IDs across the workload, scheduler, and waiter.
Slurm must report `COMPLETED` with exit `0:0`, zero restarts, and a successful
waiter. Missing or inconsistent evidence fails CI.

The `rocm-slurm-RUN_ID-ATTEMPT` Actions artifact includes:

- Rendered Dockerfiles, runtime/test build logs, and `image-build.json` with both
  image IDs and runtime sanity results.
- `image-manifest.json`, model and provenance manifests, JUnit results, and
  selected service logs.
- `client-result.json`, `controller-result.json`, `wait-result.json`,
  `terminal.json`, and `completed.json` for the final verdict.
- `collection-report.json`, which records any size-based collection omissions.

Images, model weights, and SSH keys are excluded from uploaded artifacts. The
allocation removes its run-scoped image during cleanup. The shared pinned-model
cache is retained for subsequent runs.
