# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VOXCPM_MODEL_PATH=/models/VoxCPM2 \
    VOXCPM_PROFILE=cpu \
    VOXCPM_DEVICE=cpu \
    VOXCPM_HOST=0.0.0.0 \
    VOXCPM_PORT=8090 \
    VOXCPM_REQUIRE_AUTH=1 \
    VOXCPM_ALLOW_UNAUTHENTICATED_EXTERNAL=0 \
    VOXCPM_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       ffmpeg \
       git \
       libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e '.[api]'

FROM base AS contract-smoke

RUN useradd --create-home --uid 10001 voxcpm \
    && mkdir -p /models/VoxCPM2 \
    && chown -R voxcpm:voxcpm /models /app

USER voxcpm

EXPOSE 8090

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=2).read()" || exit 1

CMD ["python", "-m", "uvicorn", "deploy.docker.smoke_app:app", "--host", "0.0.0.0", "--port", "8090"]

FROM base AS runtime

ARG VOXCPM_UPSTREAM_COMMIT=f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && git clone --filter=blob:none https://github.com/OpenBMB/VoxCPM.git /opt/VoxCPM \
    && git -C /opt/VoxCPM checkout --detach "${VOXCPM_UPSTREAM_COMMIT}" \
    && python -m pip install /opt/VoxCPM \
    && python -m pip check

RUN useradd --create-home --uid 10001 voxcpm \
    && mkdir -p /models/VoxCPM2 \
    && chown -R voxcpm:voxcpm /models /app /opt/VoxCPM

USER voxcpm

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=3).read()" || exit 1

CMD ["voxcpm-serve"]
