# PRT serving images. Two stages:
#
#   serving  stock vLLM nightly wheel of one main commit (VLLM_COMMIT, which
#            build.sh resolves from the current nightly). No source patches:
#            tuning is config-level only (config.toml), and the image adds
#            only startup packaging (FlashInfer JIT prebuild, bytecode).
#            Shared by every deployment, so deployments differ only in
#            the cache layer. Source-level vLLM optimizations live outside
#            this repo and build FROM this stage (Bo, 2026-10-07).
#   lmcache  serving + the LMCache under test, built from the `lmcache_src`
#            context. Rebuilt per tested LMCache commit. Runs both
#            `vllm serve` and `lmcache server`, so connector and server are
#            the same build.
#   mooncake serving + the Mooncake nightly wheel (GitHub nightly.yml) for
#            vLLM's MooncakeStoreConnector; also runs mooncake_master and
#            mooncake_client.
#
# Build with build.sh (it records the digests the config needs).

ARG CUDA_VERSION=13.0
ARG UBUNTU_VERSION=24.04
ARG NGC_VERSION=25.09
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:${NGC_VERSION}-cuda${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# The lmcache and mooncake stages build FROM this: the `serving` stage by default, or an
# already built serving image (build.sh passes the last-good one).
ARG SERVING_IMAGE=serving

#################### serving ####################

FROM ${BASE_IMAGE} AS serving

ARG CUDA_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        git curl patch python3 python3-dev python3-venv libxcb1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig /usr/local/cuda-${CUDA_VERSION}/compat/ \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv ~/.local/bin/uvx /usr/local/bin/ \
    && uv venv --python 3.12 /opt/venv

# Full 40-char sha; the per-commit index keeps the wheel after nightly moves on.
ARG VLLM_COMMIT
COPY constraints.txt /opt/prt/constraints.txt
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv,sharing=locked \
    test -n "${VLLM_COMMIT}" \
    && uv pip install --python /opt/venv/bin/python --prerelease=allow \
        -c /opt/prt/constraints.txt \
        'vllm[flashinfer]' grpcio grpcio-tools protobuf \
        --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cu130 \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
    && uv pip freeze --python /opt/venv/bin/python > /opt/prt/serving-freeze.txt
LABEL prt.vllm.commit=${VLLM_COMMIT}

# vLLM JITs these at every start otherwise (sampling ~86 s, trtllm_comm ~220 s,
# trtllm_utils ~7 s, fused_moe_90 ~18 min on 2026-10-07), since the workspace
# does not outlive the container. Containers run as the host user, and vLLM JITs a few small modules
# at start, so the workspace is world-writable.
ENV FLASHINFER_WORKSPACE_BASE=/opt/prt/fi_ws CUDA_HOME=/usr/local/cuda
RUN CUDA_VISIBLE_DEVICES= FLASHINFER_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 \
        /opt/venv/bin/python -c 'from flashinfer.jit.comm import gen_trtllm_comm_module; from flashinfer.jit.sampling import gen_sampling_module; from flashinfer.jit.tllm_utils import gen_trtllm_utils_module; from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module; [g().build(verbose=False, need_lock=True) for g in (gen_sampling_module, gen_trtllm_comm_module, gen_trtllm_utils_module, gen_cutlass_fused_moe_sm90_module)]' \
    && chmod -R a+rwX /opt/prt/fi_ws

# uv installs without bytecode, so every container recompiled what it imported
# (first vLLM import 9.8 s -> 6.3 s).
# The lmcache and mooncake stages inherit UV_COMPILE_BYTECODE for their installs.
RUN /opt/venv/bin/python -m compileall -q -j 32 /opt/venv/lib/python3.12/site-packages \
    > /dev/null
ENV UV_COMPILE_BYTECODE=1

WORKDIR /workspace

#################### lmcache ####################

FROM ${SERVING_IMAGE} AS lmcache

ARG MAX_JOBS=32
ENV MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST="9.0"
COPY --from=lmcache_src . /workspace/LMCache
# The serving freeze is passed as overrides, not constraints, so LMCache's caps
# never change the serving environment: every deployment runs the same vLLM
# deps and differs only in the cache layer. LMCache dev (2026-10-07) caps
# numpy, opentelemetry-api and prometheus_client below vLLM nightly's.
# gRPC codegen runs at build time with LMCache's own grpcio-tools pin, kept out
# of the venv on PYTHONPATH: the serving grpcio-tools (1.82.0rc2) emits code
# for protobuf 7, and the serving runtime is protobuf 6.
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv,sharing=locked \
    cd /workspace/LMCache \
    && uv pip install --python /opt/venv/bin/python \
        --overrides /opt/prt/serving-freeze.txt -r requirements/build.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
    && uv pip install --python /opt/venv/bin/python --no-deps --target /tmp/lmcache-codegen \
        $(grep -E '^grpcio(-tools)?==' requirements/build.txt) \
    && PYTHONPATH=/tmp/lmcache-codegen uv pip install --python /opt/venv/bin/python --no-build-isolation \
        --overrides /opt/prt/serving-freeze.txt . \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
    && rm -rf /workspace/LMCache /tmp/lmcache-codegen

#################### mooncake ####################
# The Mooncake nightly wheel, from the `mooncake_wheel` context: build.sh
# downloads it from the newest run of Mooncake's nightly.yml on GitHub, which
# builds it every night (TestPyPI only gets it when the whole nightly gate
# passes, which it has not since 2026-09-10). --no-deps: its deps come from
# PyPI under the serving freeze. ldd fails the build if a host lib is missing.
# libcuda.so.1 only exists once the NVIDIA runtime mounts it, so it is exempt
# and the import is left to the smoke test.

FROM ${SERVING_IMAGE} AS mooncake

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        libibverbs1 ibverbs-providers libnuma1 libcurl4t64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=mooncake_wheel . /opt/prt/mooncake_wheel
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv,sharing=locked \
    uv pip install --python /opt/venv/bin/python --no-deps /opt/prt/mooncake_wheel/*.whl \
    && uv pip install --python /opt/venv/bin/python \
        -c /opt/prt/serving-freeze.txt aiohttp requests msgpack \
    && P=$(/opt/venv/bin/python -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))') \
    && ! ldd "$P/mooncake_master" "$P/mooncake_client" | grep "not found" | grep -v libcuda.so.1

#################### aiperf ####################
# Load generator for every workload: AIPerf 0.13.0 + both plugins from the
# `aiperf_plugins` context, on the exact package set of the baseline venv.
# No CUDA needed.

FROM python:3.12.13-slim AS aiperf

ENV PATH="/opt/venv/bin:${PATH}"
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
COPY aiperf-constraints.txt /opt/prt/aiperf-constraints.txt
COPY --from=aiperf_plugins . /opt/prt/aiperf_plugins
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv-aiperf,sharing=locked \
    uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        -c /opt/prt/aiperf-constraints.txt aiperf==0.13.0 \
        /opt/prt/aiperf_plugins/weka_framing /opt/prt/aiperf_plugins/live_history
WORKDIR /workspace
