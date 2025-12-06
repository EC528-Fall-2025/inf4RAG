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

1. Ensure Docker Desktop (or your Docker daemon) is running.
2. Update `prometheus/prometheus-config.yaml`, replacing `host.docker.internal:8000` with the real vLLM metrics endpoint (e.g., `199.94.60.33:8000`), then save.
3. Run `./monitoring.sh start` to launch the Prometheus and Grafana containers (skip if already running).
4. In your browser, visit:
   - Prometheus targets: `http://localhost:9090/targets` (confirm the `vllm` job is `UP`)
   - Grafana login: `http://localhost:3000` (default credentials `admin/admin`; you will be prompted to change the password on first login)
5. After logging into Grafana, open Dashboards → `vLLM Overview` to view the preconfigured panels.
6. Use `./monitoring.sh status` to check container health, and `./monitoring.sh stop` when you want to shut down the stack. If your environment still relies on the legacy CLI, replace `docker compose` with `docker-compose` inside the scripts.
- Fire a test inference request against vLLM, then refresh the Grafana dashboard to confirm the charts respond.
- To swap in the official dashboard, place the JSON from the vLLM docs inside `grafana/dashboards/json/` and restart the monitoring stack.

## Validation

- Prometheus shows the `vllm` target status as `UP`.
- Grafana panels render metrics for queue time, prefill/decode durations, and token throughput.
- Any defined alerts trigger when threshold conditions are met.

[^1]: [Prometheus and Grafana for vLLM](https://docs.vllm.ai/en/v0.11.0/examples/online_serving/prometheus_grafana.html)

