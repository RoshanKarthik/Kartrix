# Kartrix server image (`kartrix serve` in hosted mode). Build: docker build -t kartrix .
# Runs as a non-root user; everything it writes goes to /data. Configuration comes from the environment —
# DATABASE_URL and the model keys are required, see deploy/README.md.

FROM python:3.12-slim

# git: checkpoints and /undo; bubblewrap: the Linux command sandbox, where the platform allows it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git bubblewrap ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# Dependencies first: this layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY kartrix ./kartrix
COPY alembic.ini ./
COPY deploy/start.sh ./start.sh
RUN uv sync --frozen --no-default-groups && chmod +x start.sh

RUN useradd --create-home --uid 10001 kartrix && mkdir -p /data && chown kartrix /data
USER kartrix

# Hosted defaults (override in the platform's environment settings).
ENV PATH="/app/.venv/bin:$PATH" \
    KARTRIX_HOME=/data/home \
    KARTRIX_SERVER__DATA_DIR=/data/server \
    KARTRIX_SERVER__HOST=0.0.0.0 \
    KARTRIX_SERVER__ALLOW_LOCAL_PATHS=false \
    KARTRIX_SERVER__ALLOW_SIGNUP=false \
    KARTRIX_SERVER__MAX_WORKERS=1 \
    KARTRIX_SEMANTIC_CACHE__ENABLED=false \
    KARTRIX_LOGGING__FILE=/data/logs/kartrix.jsonl \
    KARTRIX_DATABASE__POOLING=false

EXPOSE 8765
CMD ["/app/start.sh"]
