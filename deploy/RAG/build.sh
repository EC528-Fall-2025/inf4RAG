#!/bin/bash

set -e

root_dir=$(pwd)

# Build image for RAG deployment
podman build -t inf4rag/rag:latest -f $root_dir/RAG/Dockerfile .