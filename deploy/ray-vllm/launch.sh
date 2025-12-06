#!/bin/bash

# Ray Serve + vLLM Launch Script
# Used to launch multiple vLLM instances on OpenStack instances

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load configuration from config.yaml
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE..."
    # Source config values from read_config.sh
    eval "$(bash "$SCRIPT_DIR/read_config.sh" "$CONFIG_FILE")"
else
    echo "Warning: Config file not found: $CONFIG_FILE, using defaults"
    MODEL_PATH="gpt2"
    NUM_REPLICAS=""
    GPUS_PER_REPLICA=""
    TENSOR_PARALLEL_SIZE=""
    HOST="0.0.0.0"
    PORT="8000"
fi

# Allow environment variables and command line to override config.yaml
AUTO_DETECT="${AUTO_DETECT:-true}"  # Enable auto-detect by default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-replicas)
            NUM_REPLICAS="$2"
            shift 2
            ;;
        --gpus-per-replica)
            GPUS_PER_REPLICA="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --auto-detect)
            AUTO_DETECT="true"
            shift
            ;;
        --no-auto-detect)
            AUTO_DETECT="false"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-path PATH          Model path: Override config.yaml (default: from config.yaml)"
            echo "  --num-replicas N           Number of replicas (override config.yaml, auto-detect if not specified)"
            echo "  --gpus-per-replica N       Number of GPUs per replica (override config.yaml, auto-detect if not specified)"
            echo "  --tensor-parallel-size N  Tensor parallel size (override config.yaml, auto-detect if not specified)"
            echo "  --host HOST                Service listening address (override config.yaml)"
            echo "  --port PORT                Service port (override config.yaml)"
            echo "  --auto-detect              Enable auto-detection of GPU configuration (default: enabled)"
            echo "  --no-auto-detect           Disable auto-detection"
            echo ""
            echo "Configuration:"
            echo "  All settings can be configured in config.yaml"
            echo "  Command line arguments override config.yaml values"
            echo "  Environment variables also override config.yaml"
            echo ""
            echo "Note: If num-replicas or gpus-per-replica is not in config.yaml, the script will"
            echo "      automatically detect available GPUs and configure the deployment accordingly."
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help to see help information"
            exit 1
            ;;
    esac
done

# Check model path (skip check for HuggingFace model names)
# HuggingFace model names don't have file paths, so we only check if it's a local path
if [[ "$MODEL_PATH" == /* ]] && [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Local model path does not exist: $MODEL_PATH"
    echo "Please ensure the model path is correct, or use a HuggingFace model name (e.g., gpt2)"
    exit 1
fi

# Check GPU (only if not auto-detecting)
if [ "$AUTO_DETECT" = "true" ] && [ -z "$NUM_REPLICAS" ] && [ -z "$GPUS_PER_REPLICA" ]; then
    echo "Auto-detection enabled, GPU configuration will be determined automatically"
elif command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ -n "$NUM_REPLICAS" ] && [ -n "$GPUS_PER_REPLICA" ]; then
        TOTAL_GPUS_NEEDED=$((NUM_REPLICAS * GPUS_PER_REPLICA))
        echo "Detected $GPU_COUNT GPUs"
        echo "Need $TOTAL_GPUS_NEEDED GPUs ($NUM_REPLICAS replicas × $GPUS_PER_REPLICA GPUs/replica)"
        
        if [ $TOTAL_GPUS_NEEDED -gt $GPU_COUNT ]; then
            echo "Warning: Required GPU count exceeds available GPUs"
            echo "Ray will handle this automatically, but may not start all replicas"
        fi
    else
        echo "Detected $GPU_COUNT GPUs (will be auto-configured)"
    fi
else
    echo "Warning: nvidia-smi not detected, GPU count will be detected by Python script"
fi

# Check Python and dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

echo "Checking dependencies..."
if ! python3 -c "import ray" 2>/dev/null; then
    echo "Error: Ray not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Start service
echo "=========================================="
echo "Starting Ray Serve + vLLM"
echo "=========================================="
echo "Model path: $MODEL_PATH"
if [ -n "$NUM_REPLICAS" ]; then
    echo "Number of replicas: $NUM_REPLICAS (manual)"
else
    echo "Number of replicas: Auto-detect"
fi
if [ -n "$GPUS_PER_REPLICA" ]; then
    echo "GPUs per replica: $GPUS_PER_REPLICA (manual)"
else
    echo "GPUs per replica: Auto-detect"
fi
if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
    echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE (manual)"
else
    echo "Tensor parallel size: Auto-detect"
fi
echo "Service address: http://$HOST:$PORT"
echo "=========================================="

# Build command arguments
CMD_ARGS=(
    --model-path "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
)

if [ "$AUTO_DETECT" = "true" ]; then
    CMD_ARGS+=(--auto-detect)
fi

if [ -n "$NUM_REPLICAS" ]; then
    CMD_ARGS+=(--num-replicas "$NUM_REPLICAS")
fi

if [ -n "$GPUS_PER_REPLICA" ]; then
    CMD_ARGS+=(--gpus-per-replica "$GPUS_PER_REPLICA")
fi

if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
    CMD_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi

python3 ray_vllm_serve.py "${CMD_ARGS[@]}"

