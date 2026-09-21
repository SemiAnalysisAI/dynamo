{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/vllm_rocm_runtime.Dockerfile ===
# Initial ROCm runtime for single-node aggregate vLLM serving. This target
# does not qualify native NIXL, KVBM, GPU memory service, or media extensions.
FROM ${RUNTIME_IMAGE} AS runtime
LABEL org.opencontainers.image.base.name="{{ context.vllm.rocm.runtime_image }}"

USER root
WORKDIR /workspace
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-venv libcairo2 libnuma1 libudev1 libssl3 libjemalloc2 curl jq && \
    rm -rf /var/lib/apt/lists/*

COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /opt/uv/bin/ /opt/uv/bin/
COPY --from=wheel_builder /opt/dynamo/dist/ /opt/dynamo/dist/
COPY --from=wheel_builder /opt/dynamo/build-toolchain.json /opt/dynamo/build-toolchain.json
COPY container/deps/vllm/requirements.rocm.txt /opt/dynamo/requirements.rocm.txt
COPY container/deps/vllm/rocm_stack.py /opt/dynamo/rocm_stack.py

# Inherit the base's Torch/vLLM packages while keeping Dynamo in its own venv.
# Protect the accelerator stack in both this solve and Dockerfile.test's solve.
RUN python3 /opt/dynamo/rocm_stack.py record && \
    python3 -m venv --system-site-packages /opt/dynamo/venv && \
    /opt/dynamo/venv/bin/python3 -m pip install \
        --constraint /opt/dynamo/rocm-constraints.txt \
        --requirement /opt/dynamo/requirements.rocm.txt \
        /opt/dynamo/dist/*.whl vllm && \
    /opt/dynamo/venv/bin/python3 /opt/dynamo/rocm_stack.py verify && \
    /opt/dynamo/venv/bin/python3 -m pip check

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/opt/uv/bin:/usr/local/bin/etcd:${PATH} \
    DYNAMO_DEVICE=rocm \
    DYNAMO_HOME=/workspace

# Match the standard runtime user and allow arbitrary OpenShift UIDs in group 0.
RUN if id ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi && \
    useradd -u 1000 -m -s /bin/bash -g 0 dynamo && \
    mkdir -p /home/dynamo/.cache /opt/dynamo && \
    chown -R dynamo:0 /home/dynamo && \
    chown dynamo:0 /workspace /opt/dynamo && \
    chmod -R g+rwX /home/dynamo && \
    chmod g+rwx /workspace /opt/dynamo
ENV HOME=/home/dynamo

COPY --chmod=664 --chown=dynamo:0 LICENSE README.md Cargo.toml /workspace/
COPY --chmod=775 --chown=dynamo:0 tests/ /workspace/tests/
COPY --chmod=775 --chown=dynamo:0 examples/ /workspace/examples/
COPY --chmod=775 --chown=dynamo:0 dev/ /workspace/dev/
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common/ /workspace/components/src/dynamo/common/
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend/ /workspace/components/src/dynamo/frontend/
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/vllm/ /workspace/components/src/dynamo/vllm/
COPY --chown=dynamo:0 lib/ /workspace/lib/

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}
USER dynamo
ENTRYPOINT []
# Enroot's dockerd importer creates a container without supplying a command.
CMD ["/bin/bash"]
# No release compliance stages: this ROCm target is an opt-in qualification image.
