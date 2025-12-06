#!/bin/bash

set -e

# This script should be executed from the repository root, e.g.:
#   bash deploy/WebChat/build.sh
root_dir=$(pwd)

# Build image for the WebChat UI
podman build -t inf4rag/webchat:latest \
  -f "$root_dir/chatbotbasic/WebChat/Dockerfile" \
  "$root_dir/chatbotbasic/WebChat"
