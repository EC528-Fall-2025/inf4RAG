# PD (Prefill/Decode) Disaggregation Benchmarking Guide

## Overview

PD Disaggregation splits vLLM inference into two separate stages:
- **Prefill (Producer)**: Processes input tokens and generates KV cache on GPU 0
- **Decode (Consumer)**: Loads KV cache and generates output tokens on GPU 1

This architecture requires a proxy server to coordinate the two-stage workflow.

## Architecture

```
Client → Proxy (port 8000) → Producer (port 8100, GPU 0) → KV Cache Storage (/mnt/pd_kv)
                           ↓
                           → Consumer (port 8200, GPU 1) → Response
```

## Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install quart aiohttp
   ```

2. **Shared KV Cache Storage**:
   - Ensure `/mnt/pd_kv` directory exists and is writable
   - Both producer and consumer must have access to this path

3. **Two GPUs**:
   - GPU 0: For prefill operations
   - GPU 1: For decode operations

## Setup Instructions

### Step 1: Start Producer Server (Terminal 1)

```bash
vllm serve /mnt/models/Qwen3-4B-Instruct-2507   --host 0.0.0.0   --port 8100   --max-model-len 8192   --gpu-memory-utilization 0.8   --trust-remote-code   --enforce-eager   --kv-transfer-config '{
    "engine_id": "my-disagg-engine-1",
    "kv_connector": "SharedStorageConnector",
    "kv_role": "kv_producer",
    "kv_parallel_size": 1,
    "kv_buffer_size": 1e9,
    "kv_connector_extra_config": {
      "shared_storage_path": "/mnt/pd_kv"
    }
  }'
```

**Key Parameters**:
- `--port 8100`: Producer listens on port 8100
- `--kv-role kv_producer`: Sets server as KV cache producer
- `--kv-storage-connector-path /mnt/pd_kv`: Shared storage location

### Step 2: Start Consumer Server (Terminal 2)

```bash
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=1

vllm serve /mnt/models/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8200 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --enforce-eager \
  --kv-transfer-config '{
    "engine_id": "my-disagg-engine-1",
    "kv_connector": "SharedStorageConnector",
    "kv_role": "kv_consumer",
    "kv_parallel_size": 1,
    "kv_buffer_size": 1e9,
    "kv_connector_extra_config": {
      "shared_storage_path": "/mnt/pd_kv"
    }
  }'
  
```

**Key Parameters**:
- `--port 8200`: Consumer listens on port 8200
- `--kv-role kv_consumer`: Sets server as KV cache consumer
- `--kv-storage-connector-path /mnt/pd_kv`: Shared storage location (same as producer)

### Step 3: Start Proxy Server (Terminal 3)

```bash
python pd_proxy.py \
    --port 8000 \
    --prefill-url http://localhost:8100/v1/completions \
    --decode-url http://localhost:8200/v1/completions
```

**Proxy Configuration**:
- `--port 8000`: Client-facing port
- `--prefill-url`: Producer endpoint
- `--decode-url`: Consumer endpoint
- `--rate-limit 40`: Max requests per second (optional)
- `--max-concurrent 100`: Max concurrent requests (optional)

### Step 4: Run Benchmark (Terminal 4)

```bash
# Clear KV cache for fresh benchmark (optional but recommended)
sudo rm -rf /mnt/pd_kv/*

# Run PD disaggregation benchmarks
PORT=8000 ./run_and_process_sprint4.sh pd-disaggregation
```

## How It Works

1. **Client sends request** to proxy on port 8000
2. **Proxy forwards to producer** with `max_tokens=1` (prefill stage)
   - Producer processes input tokens on GPU 0
   - Generates and saves KV cache to `/mnt/pd_kv`
3. **Proxy forwards to consumer** with original request (decode stage)
   - Consumer loads KV cache from `/mnt/pd_kv`
   - Generates output tokens on GPU 1
4. **Proxy streams response** back to client

## Verification

Monitor both GPUs to confirm proper operation:

```bash
watch -n 1 nvidia-smi
```

**Expected Behavior**:
- **GPU 0**: Utilization during prefill operations
- **GPU 1**: Utilization during decode operations
- **Terminal 1**: POST requests to `/v1/chat/completions`, "Inject KV cache" messages
- **Terminal 2**: POST requests to `/v1/chat/completions`, "External Cache Hit!" messages
- **Terminal 3**: 200 status responses (not 404s)

## Troubleshooting

### Issue: 404 Errors in Proxy Logs

**Problem**: Proxy doesn't implement required endpoints

**Solution**: Ensure `pd_proxy.py` has both routes:
```python
@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
```

### Issue: Consumer Shows 0% GPU Utilization

**Problem**: Requests going directly to producer, bypassing consumer

**Solution**: 
1. Verify proxy is running and accessible
2. Ensure benchmark targets proxy port (8000), not producer port (8100)
3. Check proxy logs for successful request forwarding

### Issue: "External Cache Hit!" But No GPU Work

**Problem**: KV cache from previous runs causing immediate hits

**Solution**: Clear cache before benchmarking:
```bash
sudo rm -rf /mnt/pd_kv/*
```

### Issue: Missing Dependencies

**Problem**: `ModuleNotFoundError: No module named 'rate_limiter'`

**Solution**: Ensure all three files are present:
- `pd_proxy.py`
- `rate_limiter.py`
- `request_queue.py`

## Files

- **`pd_proxy.py`**: Main proxy server with request routing logic
- **`rate_limiter.py`**: Token bucket rate limiting implementation
- **`request_queue.py`**: Concurrent request queue management
- **`run_and_process_sprint4.sh`**: Benchmark execution script

## Results

Benchmark results are automatically:
1. Archived to `sprint4_results/pd-disaggregation/`
2. Extracted and parsed by `process_sprint4_results.py`
3. Populated into `vLLM_benchmark_EC528_-_Sprint4_RESULTS.csv` columns O-T

## Notes

- The proxy implements a two-stage workflow: prefill with `max_tokens=1`, then decode with full request
- Cache hits are expected after the first request for each unique input
- For accurate performance measurement, clear cache between test runs
- Both servers must be fully started before running benchmarks
- The proxy must be running for PD disaggregation to work - without it, requests go directly to producer and consumer is never utilized
