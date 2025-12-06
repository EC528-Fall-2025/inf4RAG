#!/bin/bash

set -e

# Run the WebChat UI container.
# The UI will be available on http://localhost:7860 by default.
# Adjust the -p mapping if you want to expose a different host port.

podman run --rm \
  -p 7860:7860 \
  inf4rag/webchat:latest
