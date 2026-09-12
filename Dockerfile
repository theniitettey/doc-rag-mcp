FROM python:3.11-slim

# curl is needed for the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV RAG_DOCS_DIR=/data/docs
# Lets `python -c "import server; ..."` (e.g. make clear-cache) work from
# any working directory in the container.
ENV PYTHONPATH=/app/src
# 0.0.0.0 is required here (not the CLI's local-machine default of
# 127.0.0.1) so Docker's port mapping can actually reach the process.
# The container boundary + RAG_AUTH_TOKEN are what keep this safe, not
# the bind address. Deliberately not 8000/5000/3000 -- picked to avoid
# clashing with other local dev servers; override via docker-compose's
# RAG_PORT (from .env) if you need a different one.
ENV RAG_HOST=0.0.0.0
ENV RAG_PORT=8743

RUN mkdir -p /data/docs

EXPOSE 8743

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:${RAG_PORT}/health || exit 1

CMD ["python", "src/server.py", "--transport", "http"]