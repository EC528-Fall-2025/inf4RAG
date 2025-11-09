#!/usr/bin/env sh
set -euo pipefail

: "${PORT:=8001}"
: "${RAG_PERSIST_DIR:=/data}"
: "${WORKERS:=2}"
: "${THREADS:=4}"

mkdir -p "$RAG_PERSIST_DIR"
echo "Starting RAG on port ${PORT}, data dir ${RAG_PERSIST_DIR}"

exec gunicorn -w "$WORKERS" -k gthread --threads "$THREADS" -b "0.0.0.0:${PORT}" entrypoint:app
