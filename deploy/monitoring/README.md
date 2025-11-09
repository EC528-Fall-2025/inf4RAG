# vLLM Monitoring Stack

This module bundles Prometheus and Grafana for tracking vLLM metrics such as request latency, queue depth, and token throughput. It follows the official vLLM monitoring example.[^1]

## Layout

- `docker-compose.yaml` – orchestrates Prometheus and Grafana containers with persistent volumes.
- `prometheus/prometheus-config.yaml` – scrape configuration targeting the vLLM metrics endpoint.
- `grafana/` – provisioning assets: Prometheus datasource and a prebuilt vLLM dashboard JSON.
- `start.sh`, `stop.sh`, `status.sh` – helper scripts for operating the stack (or use `monitoring.sh start|stop|status` for a single entry point).

## Prerequisites

- Docker Engine and Docker Compose.
- vLLM HTTP server exposing `/metrics` (default `http://host.docker.internal:8000/metrics`). Update the scrape target in `prometheus/prometheus-config.yaml` if your host/port differ.
- Optional host volumes if you want to persist Grafana data outside the containers.

## Quick Start

1. Adjust the scrape target in `prometheus/prometheus-config.yaml` if necessary.
2. Run `./start.sh` (or `./monitoring.sh start`) to launch the stack in the background.
3. Visit Prometheus at `http://localhost:9090` and Grafana at `http://localhost:3000` (defaults: admin/admin).
4. Grafana auto-loads:
   - Datasource `Prometheus` (points to `http://prometheus:9090` inside the compose network).
   - Dashboard `vLLM Overview` from `grafana/dashboards/json/vllm-overview.json`.
   If you prefer the official dashboard, replace the JSON in that folder with the one from the docs[^1].
5. Use `./status.sh` (or `./monitoring.sh status`) to inspect container health and `./stop.sh` (or `./monitoring.sh stop`) to stop the stack.
- If your environment still uses the legacy `docker-compose` CLI, replace the `docker compose` commands in the scripts with `docker-compose`.
- Prometheus shows the `vllm` target status as `UP`.
- Grafana panels render metrics for queue time, prefill/decode durations, and token throughput.
- Any defined alerts trigger when threshold conditions are met.
- Issue one or two inference requests to the vLLM service, then refresh the Grafana dashboard to confirm the metrics respond accordingly.

## Validation

- Prometheus shows the `vllm` target status as `UP`.
- Grafana panels render metrics for queue time, prefill/decode durations, and token throughput.
- Any defined alerts trigger when threshold conditions are met.

[^1]: [Prometheus and Grafana for vLLM](https://docs.vllm.ai/en/v0.11.0/examples/online_serving/prometheus_grafana.html)

