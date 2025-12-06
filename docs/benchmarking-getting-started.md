# Benchmarking Guide

## Overview
The `run_and_process.sh` script provides an automated benchmarking workflow for comparing different vLLM parallelism strategies and configurations. It handles both benchmark execution and results processing in a single command.

## What It Does
The script automates the following workflow:
1. **Runs Steady Benchmarks**: Executes steady-state load tests at multiple request rates (2, 4, 8, 16, 32 req/s) for 90 seconds each
2. **Runs Flood Benchmark**: Executes a flood test where all requests arrive simultaneously to measure peak performance
3. **Processes Results**: Automatically parses benchmark output and populates metrics into the results CSV
4. **Archives Data**: Creates compressed archives of all benchmark results for reproducibility

## Supported Experiment Types
The script supports benchmarking the following parallelism strategies:
- **tensor-parallelism**: Model distributed across GPUs using tensor parallelism
- **pipeline-parallelism**: Model distributed using pipeline parallelism
- **pd-disaggregation**: Prefill-Decode disaggregation with shared storage
- **pd-disaggregation-p2p**: Prefill-Decode disaggregation using P2P NCCL connector
- **data-parallelism**: Multiple replicas of the model for parallel request processing

## Usage

### Basic Usage:
```bash
cd benchmark-suite
./run_and_process.sh <experiment_set_name>
```

### Examples:
```bash
# Benchmark tensor parallelism configuration
./run_and_process.sh tensor-parallelism

PORT=10001 ./run_and_process.sh pd-disaggregation-p2p
```

### Custom Configuration:
You can override default settings using environment variables:
```bash
# Custom port and duration
PORT=8100 DURATION=120 ./run_and_process.sh tensor-parallelism

# Custom request rates
REQUEST_RATES="5 10 20" ./run_and_process.sh data-parallelism
```

## Environment Variables
- `PORT` (default: 8000): Port where the vLLM server is running
- `HOST` (default: 127.0.0.1): Hostname of the vLLM server
- `DURATION` (default: 90): Duration in seconds for each steady benchmark
- `REQUEST_RATES` (default: "2 4 8 16 32"): Space-separated list of request rates to test
- `MODEL_TYPE` (default: "chat"): Model type (chat or completions)

## Output
Results are organized as follows:
```
benchmark-suite/
├── sprint4_results/
│   ├── tensor-parallelism/
│   │   ├── bench_*.tar.gz           # Archived results for each test
│   ├── pd-disaggregation-p2p/
│   │   ├── bench_*.tar.gz
│   └── ...
└── vLLM_benchmark_EC528_-_Sprint4_RESULTS.csv  # Consolidated results
```

## Metrics Collected
For each benchmark configuration, the following metrics are collected:
- **TTFT (Time to First Token)**: Latency until the first token is generated
- **TPOT (Time Per Output Token)**: Average time to generate each subsequent token
- **ITL (Inter-Token Latency)**: Latency between consecutive tokens

Each metric includes Mean, Median, and P99 statistics across all requests.

## Prerequisites
Before running benchmarks, ensure:
1. vLLM server is running and accessible at the specified HOST:PORT
2. The model is properly loaded and ready to serve requests
3. You have sufficient disk space for result archives
4. Python dependencies are installed: `pip install pandas click`

## Troubleshooting
- **"Model name fetch failed"**: Ensure the vLLM server is running and the `/v1/models` endpoint is accessible
- **"Column not found" errors**: Verify your CSV template has the correct headers for the experiment type
- **Benchmark timeouts**: Increase `DURATION` or reduce `REQUEST_RATES` for slower configurations
