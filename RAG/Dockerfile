# syntax=docker/dockerfile:1.7
# Lightweight production image for the Flask RAG service

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Set envs for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install build deps only if needed (none for current requirements, keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m rag && mkdir -p /app /data && chown -R rag:rag /app /data
WORKDIR /app

# Copy requirements and install (pin if desired)
COPY RAG/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy source code
COPY RAG/ .

# Switch to non-root
USER rag

# Expose port
EXPOSE 8001

# Health metadata (optional)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:${PORT:-8001}/rag/health || exit 1

# Default environment
ENV PORT=8001 \
    RAG_PERSIST_DIR=/data \
    RAG_DEBUG=0

# Entrypoint (gunicorn for production)
# entrypoint:app points to Flask application instance
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:8001", "entrypoint:app"]
