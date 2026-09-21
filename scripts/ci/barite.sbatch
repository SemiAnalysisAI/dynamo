#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Resource limits are supplied by the trusted submission controller.
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP
unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTEST_DISABLE_PLUGIN_AUTOLOAD
export PYTHONNOUSERSITE=1
RUN_DIR=${1:?run directory required}
CONTROLLER="$RUN_DIR/controller"
ROOT=$(dirname "$(dirname "$RUN_DIR")")
child=''
scratch=''
finished=0

# Wait explicitly: a foreground srun can defer shell traps until it exits.
run_step() {
    "$@" &
    child=$!
    local result=0
    wait "$child" || result=$?
    child=''
    return "$result"
}
cleanup() {
    local result=$?
    trap - EXIT TERM INT USR1
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        # srun forwards TERM to its step; Slurm's allocation timeout remains the backstop.
    fi
    if [[ "$finished" != 1 ]]; then
        python3 - "$RUN_DIR" "$result" <<'PY'
import json, os, pathlib, sys, time
run = pathlib.Path(sys.argv[1])
request = json.loads((run / 'request.json').read_text())
result = {'schema_version': 1, 'status': 'failed', 'run_key': request['run_key'],
          'source_sha': request['source_sha'], 'job_id': os.environ.get('SLURM_JOB_ID'),
          'exit_code': int(sys.argv[2]), 'ended_at': time.time()}
path = run / 'completed.json.tmp'
path.write_text(json.dumps(result) + '\n')
os.replace(path, run / 'completed.json')
PY
    fi
    if [[ -n "$scratch" && "$scratch" == /tmp/dynamo-rocm-* ]]; then
        rm -rf -- "$scratch"
    fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 124' USR1

: "${SLURM_JOB_ID:?must run inside a Slurm allocation}"
: "${SLURMD_NODENAME:?must execute on slurmd compute node}"
: "${SLURM_JOB_NODELIST:?missing allocated node list}"
: "${ROCR_VISIBLE_DEVICES:?missing Slurm GPU assignment}"
[[ $(hostname -s) == "$SLURMD_NODENAME" ]]
scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -Fx -- "$SLURMD_NODENAME" >/dev/null
[[ ${SLURM_RESTART_COUNT:-0} == 0 ]]
[[ ${SLURM_CPUS_PER_TASK:-0} == 16 ]]
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
python3 - "$RUN_DIR" <<'PY'
import json, os, pathlib, sys
run = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(run / 'controller'))
from barite_common import verify_controller, atomic_json
verify_controller(run)
atomic_json(run / 'allocation.json', {k: os.environ.get(k) for k in (
    'SLURM_JOB_ID','SLURMD_NODENAME','SLURM_JOB_NODELIST','SLURM_CPUS_PER_TASK',
    'SLURM_JOB_GPUS','SLURM_RESTART_COUNT','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES')})
PY
scratch=$(mktemp -d "/tmp/dynamo-rocm-${SLURM_JOB_ID}-XXXXXXXX")
mkdir -p "$ROOT/images" "$ROOT/models"
python3 "$CONTROLLER/barite_verify.py" extract --run-dir "$RUN_DIR" --destination "$scratch/source"
JOB_SOURCE="$scratch/source"
readarray_compat() {
    python3 - "$RUN_DIR/request.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['source_sha']); print(r['archive_sha256']); print(r.get('reuse_image_sha') or '')
PY
}
mapfile -t identity < <(readarray_compat)
SOURCE_SHA=${identity[0]}
ARCHIVE_SHA=${identity[1]}
REUSE_SHA=${identity[2]:-}
BASE_URI='docker://registry-1.docker.io#vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1'
COMMON=(srun --ntasks=1 --cpu-bind=cores --kill-on-bad-exit=1 --no-container-mount-home --no-container-entrypoint)
if [[ -z "$REUSE_SHA" ]]; then
    BUILD_SQSH="$RUN_DIR/dynamo-test.building.sqsh"
    [[ ! -e "$BUILD_SQSH" ]]
    run_step "${COMMON[@]}" --container-image="$BASE_URI" --container-remap-root --container-writable \
        --container-env=ROCR_VISIBLE_DEVICES --container-mounts="$JOB_SOURCE:/workspace,$RUN_DIR:/results" \
        --container-workdir=/workspace --container-save="$BUILD_SQSH" \
        bash /results/controller/rocm/install-source.sh --source-sha "$SOURCE_SHA" \
        --archive-sha256 "$ARCHIVE_SHA" --base-uri "$BASE_URI" --manifest /results/build-manifest.json
    [[ -s "$BUILD_SQSH" ]]
    python3 "$CONTROLLER/barite_verify.py" image-manifest --run-dir "$RUN_DIR" --image "$BUILD_SQSH" \
        --build-manifest "$RUN_DIR/build-manifest.json" --output "$RUN_DIR/image-manifest.json"
    # Content-addressed publication uses an atomic directory rename, never overwrites.
    TEST_SQSH=$(python3 - "$RUN_DIR" "$ROOT/images" <<'PY'
import json, os, pathlib, shutil, sys, tempfile
run, root = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((run / 'image-manifest.json').read_text())
digest = manifest['sqsh_sha256']
import re
if not re.fullmatch('[0-9a-f]{64}', digest): raise ValueError('invalid image digest')
target = root / digest
staging = pathlib.Path(tempfile.mkdtemp(prefix='.publishing-', dir=root))
try:
    os.rename(run / 'dynamo-test.building.sqsh', staging / 'image.sqsh')
    shutil.copyfile(run / 'image-manifest.json', staging / 'image-manifest.json')
    try: os.rename(staging, target)
    except OSError:
        if not target.is_dir(): raise
        # Existing artifacts are validated by verify-image immediately below.
finally:
    if staging.exists(): shutil.rmtree(staging)
print(target / 'image.sqsh')
PY
)
else
    TEST_SQSH="$ROOT/images/$REUSE_SHA/image.sqsh"
    cp "$ROOT/images/$REUSE_SHA/image-manifest.json" "$RUN_DIR/image-manifest.json"
fi
python3 "$CONTROLLER/barite_verify.py" verify-image --run-dir "$RUN_DIR" --image "$TEST_SQSH" \
    --manifest "$RUN_DIR/image-manifest.json"
# Prepare the pinned model once in an allocated container, then remount read-only.
run_step "${COMMON[@]}" --container-image="$TEST_SQSH" --container-env=ROCR_VISIBLE_DEVICES \
    --container-mounts="$RUN_DIR:/results,$ROOT/models:/model-store" \
    --container-workdir=/opt/dynamo \
    /opt/dynamo/venv/bin/python3 /results/controller/rocm/prepare-models.py \
    --cache-root /model-store --output /results/model-manifest.json
MODEL_KEY=$(python3 - "$RUN_DIR/model-manifest.json" <<'PY'
import json,re,sys
key=json.load(open(sys.argv[1]))['cache_key']
if not re.fullmatch('[0-9a-f]{64}',key): raise ValueError('invalid model cache key')
print(key)
PY
)
MODEL_CACHE="$ROOT/models/$MODEL_KEY"
[[ -d "$MODEL_CACHE" ]]
export HF_HOME=/models HF_HUB_CACHE=/models/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run_step "${COMMON[@]}" --container-image="$TEST_SQSH" \
    --container-env=ROCR_VISIBLE_DEVICES,HF_HOME,HF_HUB_CACHE,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE \
    --container-mounts="$JOB_SOURCE:/workspace,$RUN_DIR:/results,$MODEL_CACHE:/models:ro" \
    --container-workdir=/workspace \
    bash /results/controller/rocm/run-tests.sh --suite aggregate --python /opt/dynamo/venv/bin/python3 --results /results
python3 "$CONTROLLER/barite_verify.py" workload --run-dir "$RUN_DIR" --runtime-dir "$RUN_DIR" --output "$RUN_DIR/completed.json"
finished=1
