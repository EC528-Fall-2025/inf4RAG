# syntax=docker/dockerfile:1.7
# Lightweight image for the Gradio WebChat UI

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Basic Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1     \
    PIP_NO_CACHE_DIR=on    \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Create working directory
WORKDIR /app

# Install minimal OS dependencies (certificates, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Gradio listens on this port by default (see app.py)
EXPOSE 7860

# Default environment – you can override these in podman/docker run
ENV GRADIO_SERVER_PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    WEBCHAT_CONFIG=/app/config.yaml

# Start the WebChat app
CMD ["python", "app.py"]

