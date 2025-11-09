#!/bin/bash
#
# run_and_process_sprint4.sh
#
# A comprehensive script to BOTH run a vLLM benchmark set AND process the results.
# It automates the entire workflow for a single experiment configuration.
#
# Usage:
#   ./run_and_process_sprint4.sh <experiment_set_name> [path_to_sheet.csv]
#
# Arguments:
#   - experiment_set_name: (Required) The name for the set, which must match a key
#     in the processing script.
#     Examples: tensor-parallelism, pipeline-parallelism, pd-disaggregation
#
#   - path_to_sheet.csv: (Optional) Path to the CSV file to be updated.
#     Defaults to 'vLLM benchmark EC528 - Sprint4.csv' in the same directory.
#
# Example:
#   # This will run the tensor-parallelism benchmarks and update the default CSV.
#   ./run_and_process_sprint4.sh tensor-parallelism

# --- Default Settings (General, with Sprint 4 specifics) ---
set -e # Exit immediately if a command exits with a non-zero status.

# Benchmark parameters
DURATION=${DURATION:-90}
REQUEST_RATES=${REQUEST_RATES:-"2 4 8 16 32"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8000}
MODEL_TYPE=${MODEL_TYPE:-"chat"}

# File and directory settings
BASE_RESULTS_DIR="sprint4_results"
DEFAULT_SHEET_NAME="vLLM benchmark EC528 - Sprint4.csv"
RESULTS_SHEET_NAME="vLLM_benchmark_EC528_-_Sprint4_RESULTS.csv"

# --- Script Logic ---

# 1. Argument Validation
if [ -z "$1" ]; then
  echo "ERROR: Missing experiment set name."
  echo "Usage: $0 <experiment_set_name> [path_to_sheet.csv]"
  echo "Example: $0 tensor-parallelism"
  exit 1
fi

EXPERIMENT_SET_NAME=$1
# Use the provided sheet path or the default
SHEET_ARG=${2:-$DEFAULT_SHEET_NAME}

# 2. Resolve Paths
# Get the absolute path to the directory containing this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
BENCH_SCRIPT_PATH="${SCRIPT_DIR}/bench.py"
PROCESS_SCRIPT_PATH="${SCRIPT_DIR}/process_sprint4_results.py"
TARGET_DIR="${SCRIPT_DIR}/${BASE_RESULTS_DIR}/${EXPERIMENT_SET_NAME}"
SHEET_PATH="${SCRIPT_DIR}/${SHEET_ARG}"
RESULTS_PATH="${SCRIPT_DIR}/${RESULTS_SHEET_NAME}"

# If the results CSV already exists, use it as input; otherwise use the template
if [ -f "$RESULTS_PATH" ]; then
    echo "Using existing results file: ${RESULTS_PATH}"
    SHEET_PATH="$RESULTS_PATH"
fi

# Verify that the necessary scripts and sheet exist
if [ ! -f "$BENCH_SCRIPT_PATH" ]; then
    echo "ERROR: Benchmark script not found at ${BENCH_SCRIPT_PATH}" >&2
    exit 1
fi
if [ ! -f "$PROCESS_SCRIPT_PATH" ]; then
    echo "ERROR: Processing script not found at ${PROCESS_SCRIPT_PATH}" >&2
    exit 1
fi
if [ ! -f "$SHEET_PATH" ]; then
    echo "ERROR: Benchmark sheet not found at ${SHEET_PATH}" >&2
    echo "Please provide the path as the second argument if it's not in the default location." >&2
    exit 1
fi

# 3. Set up directories and start benchmarks
echo "---"
echo "Starting Full Run for Benchmark Set: ${EXPERIMENT_SET_NAME}"
echo "Results will be archived in: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"
echo "---"

# The bench.py script creates archives in its current working directory.
# We cd into the target directory to keep results organized.
cd "${TARGET_DIR}" || exit 1

# 4. Run Steady Benchmarks
echo "--- Running STEADY benchmarks ---"
for rate in $REQUEST_RATES; do
  echo ""
  echo "==> Starting steady test with request rate: ${rate} req/s for ${DURATION}s <=="
  python3 "$BENCH_SCRIPT_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --model-type "$MODEL_TYPE" \
    steady --request-rate "$rate" --duration "$DURATION"
done
echo "--- STEADY benchmarks complete ---"
echo ""

# 5. Run Flood Benchmark
echo "--- Running FLOOD benchmark ---"
python3 "$BENCH_SCRIPT_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --model-type "$MODEL_TYPE" \
  flood
echo "--- FLOOD benchmark complete ---"
echo ""

# Return to the original script directory to run the processing script
cd "$SCRIPT_DIR" > /dev/null

# 6. Process the results automatically
echo "---"
echo "Benchmark runs complete. Now processing results..."
echo "---"

python3 "$PROCESS_SCRIPT_PATH" "$TARGET_DIR" "$SHEET_PATH"

# 7. Final message
echo "---"
echo "Full run for '${EXPERIMENT_SET_NAME}' has finished."
echo "Result archives are in: ${TARGET_DIR}"
echo "Updated sheet is at: ${SHEET_PATH}"
echo "---"
