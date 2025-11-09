#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage: ./monitoring.sh <command>

Commands:
  start    Launch Prometheus and Grafana containers (detached)
  stop     Stop and remove the monitoring containers
  status   Show current container status
EOF
}

cmd="${1:-help}"

case "$cmd" in
  start)
    docker compose up -d
    ;;
  stop)
    docker compose down
    ;;
  status)
    docker compose ps
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac

