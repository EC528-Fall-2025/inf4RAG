#!/usr/bin

set -e

mkdir -p $(pwd)/persistent_data

podman run --rm -p 8001:8001 \
	-e PORT=8001 \
	-e RAG_PERSIST_DIR=/data \
	-v $(pwd)/persistent_data:/data:Z \
    -u root --privileged=true \
	inf4rag/rag:latest
