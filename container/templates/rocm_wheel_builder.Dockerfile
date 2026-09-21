{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/rocm_wheel_builder.Dockerfile ===
# Build image-local wheels against the same Python/ROCm base as the runtime.
# These are not portable manylinux wheels and are not published separately.
FROM dynamo_base AS wheel_builder

ARG CARGO_BUILD_JOBS
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} \
    CARGO_TARGET_DIR=/opt/dynamo/target

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential clang libclang-dev pkg-config libssl-dev cmake \
        python3-dev python3-venv libnuma-dev libudev-dev libcairo2-dev && \
    rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04's protoc is too old for the runtime's proto3 optional fields.
ADD --checksum=sha256:{{ context.vllm.rocm.protoc_sha256 }} \
    https://github.com/protocolbuffers/protobuf/releases/download/v{{ context.vllm.rocm.protoc_version }}/protoc-{{ context.vllm.rocm.protoc_version }}-linux-x86_64.zip /tmp/protoc.zip
RUN unzip -o /tmp/protoc.zip -d /opt/dynamo/protoc && \
    chmod 755 /opt/dynamo/protoc/bin/protoc && rm /tmp/protoc.zip
ENV PROTOC=/opt/dynamo/protoc/bin/protoc \
    PROTOC_INCLUDE=/opt/dynamo/protoc/include

RUN python3 -m venv --system-site-packages /opt/dynamo/build-venv && \
    /opt/dynamo/build-venv/bin/python3 -m pip install \
        maturin==1.9.6 hatchling==1.27.0 build==1.3.0 patchelf==0.17.2.4 pyyaml==6.0.3
ENV VIRTUAL_ENV=/opt/dynamo/build-venv \
    PATH=/opt/dynamo/build-venv/bin:${PATH}

# Keep the versions of the tools that produced the image-local wheels.
RUN python3 -c 'import json, os, subprocess, sys; tools = {name: subprocess.check_output([os.environ["PROTOC"] if name == "protoc" else name, "--version"], text=True).strip() for name in ("rustc", "cargo", "maturin", "protoc")}; tools["python"] = sys.version; print(json.dumps(tools, indent=2))' \
    > /opt/dynamo/build-toolchain.json

WORKDIR /workspace
COPY .cargo/ /workspace/.cargo/
COPY pyproject.toml README.md LICENSE Cargo.toml Cargo.lock rust-toolchain.toml hatch_build.py /workspace/
COPY lib/ /workspace/lib/
COPY components/ /workspace/components/
# Complete the Cargo workspace for the binding and Python build backends.
COPY examples/router/custom-policy-example/ /workspace/examples/router/custom-policy-example/
COPY deploy/inference-gateway/ext-proc/ /workspace/deploy/inference-gateway/ext-proc/
COPY deploy/inference-gateway/sidecar/ /workspace/deploy/inference-gateway/sidecar/

# Use nixl-sys's upstream dlopen fallback. Native NIXL/KVBM and CUDA media
# features are not supported by this initial aggregate-serving ROCm image.
# CUDARC_CUDA_VERSION selects declarations; it does not install a CUDA runtime.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=shared \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=shared \
    NIXL_PREFIX=/opt/dynamo/no-native-nixl CUDARC_CUDA_VERSION=13030 \
        python3 -m maturin build --locked --release --auditwheel skip \
        --manifest-path lib/bindings/python/Cargo.toml --out /opt/dynamo/dist && \
    python3 -m build --wheel --no-isolation --outdir /opt/dynamo/dist .

# Build the Cairo binding here so runtime dependency installation needs no compiler.
RUN python3 -m pip wheel --no-deps --wheel-dir /opt/dynamo/dist pycairo==1.28.0

COPY container/deps/requirements.aisimulate.txt /tmp/requirements.aisimulate.txt
RUN python3 -m pip download --only-binary=:all: --no-deps --no-index \
    --find-links https://pypi.nvidia.com/aisimulate/ --dest /opt/dynamo/dist \
    --requirement /tmp/requirements.aisimulate.txt
