# Multi-LexSum Interactive Case Analyzer
#
# Cross-platform image: python:3.11-slim is published as a multi-arch manifest
# (linux/amd64 + linux/arm64), so the same Dockerfile builds natively on both
# Windows (amd64) and Apple Silicon Macs (arm64) without modification.
#
# Build context is the Interactive_Tool/ directory, so .dockerignore (next to
# this file) is honored. Build from the repo root via:
#   docker build -t multilexsum-tool Interactive_Tool/
# or via the compose file:
#   docker compose -f Interactive_Tool/docker-compose.yml up --build

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    NLTK_DATA=/usr/local/share/nltk_data \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so the layer caches across code edits.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Pre-download NLTK corpora at build time so the container starts offline-ready
# (Docker Desktop on Windows/macOS can run with restricted network for the
# compute node, and we don't want first-request latency to download these).
RUN python -m nltk.downloader -d "$NLTK_DATA" \
    punkt punkt_tab stopwords wordnet omw-1.4

COPY . /app/

EXPOSE 7860

# Bind to 0.0.0.0 so the port is reachable from the host across Docker Desktop
# (Windows/WSL2) and Docker on macOS (gVisor / VM bridge).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
