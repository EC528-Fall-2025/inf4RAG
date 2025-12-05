#!/bin/bash
# Helper script to read config.yaml and output as shell variables

CONFIG_FILE="${1:-config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "MODEL_PATH=gpt2"
    echo "NUM_REPLICAS="
    echo "GPUS_PER_REPLICA="
    echo "TENSOR_PARALLEL_SIZE="
    echo "HOST=0.0.0.0"
    echo "PORT=8000"
    exit 0
fi

python3 << PYTHON_EOF
import yaml
import sys
import os

config_file = "$CONFIG_FILE"

try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    model_path = config.get('model', {}).get('path', 'gpt2')
    num_replicas = config.get('deployment', {}).get('num_replicas')
    gpus_per_replica = config.get('deployment', {}).get('gpus_per_replica')
    tensor_parallel_size = config.get('deployment', {}).get('tensor_parallel_size')
    host = config.get('service', {}).get('host', '0.0.0.0')
    port = config.get('service', {}).get('port', 8000)
    
    print(f"MODEL_PATH={model_path}")
    print(f"NUM_REPLICAS={num_replicas if num_replicas is not None else ''}")
    print(f"GPUS_PER_REPLICA={gpus_per_replica if gpus_per_replica is not None else ''}")
    print(f"TENSOR_PARALLEL_SIZE={tensor_parallel_size if tensor_parallel_size is not None else ''}")
    print(f"HOST={host}")
    print(f"PORT={port}")
except Exception as e:
    print(f"# Error reading config: {e}", file=sys.stderr)
    print("MODEL_PATH=gpt2")
    print("NUM_REPLICAS=")
    print("GPUS_PER_REPLICA=")
    print("TENSOR_PARALLEL_SIZE=")
    print("HOST=0.0.0.0")
    print("PORT=8000")
PYTHON_EOF

