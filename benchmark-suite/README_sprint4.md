# Sprint 4 Benchmarking Guide

This document outlines how to run the Sprint 4 benchmarks and process the results using a single, unified script.

## Overview

The `run_and_process_sprint4.sh` script automates the entire workflow for a single vLLM backend configuration (e.g., Tensor Parallelism). It handles both running the benchmarks and processing the results to fill out the spreadsheet.

## Prerequisites

1.  **vLLM Server**: You must have a vLLM server running and accessible. The script defaults to `127.0.0.1:8000`.
2.  **Python Environment**: Ensure you have Python 3 installed, along with the necessary packages.
    ```bash
    pip install pandas click
    ```
3.  **Benchmark Sheet**: You need the original CSV export of the Google Sheet. By default, the script looks for a file named `vLLM_benchmark_EC528_-_Sprint4.csv` in the `benchmark-suite` directory.

## Step-by-Step Instructions

### Step 1: Run the Combined Script

You will run the `run_and_process_sprint4.sh` script once for each of the three experiment sets:
-   Tensor Parallelism
-   Pipeline Parallelism
-   PD Disaggregation

Before running the script for a set, **manually configure your vLLM backend** for that specific setup (e.g., start the vLLM server with `tensor-parallel-size=2`).

**To run a full experiment set:**

Execute the script from within the `benchmark-suite` directory. Provide the name of the experiment set as an argument. This name **must match** one of the keys used for processing (`tensor-parallelism`, `pipeline-parallelism`, or `pd-disaggregation`).

```bash
# Example for the Tensor Parallelism set
./run_and_process_sprint4.sh tensor-parallelism

# Example for the Pipeline Parallelism set
./run_and_process_sprint4.sh pipeline-parallelism

# Example for the PD Disaggregation set
./run_and_process_sprint4.sh pd-disaggregation
```

### What the Script Does

1.  **Runs Benchmarks**:
    -   Creates a results directory (e.g., `sprint4_results/tensor-parallelism/`).
    -   Runs a "steady" benchmark for each request rate (2, 4, 8, 16, 32) for 90 seconds.
    -   Runs one "flood" benchmark.
    -   Saves the compressed result of each run (`.tar.gz`) into the results directory.

2.  **Processes Results**:
    -   Immediately after the benchmarks finish, it automatically calls the `process_sprint4_results.py` script.
    -   It parses all the newly created `.tar.gz` archives.
    -   It extracts the key metrics (TTFT, TPOT, ITL) and writes them into the correct cells of a **new CSV file** named `vLLM_benchmark_EC528_-_Sprint4_RESULTS.csv`.

Repeat this process for all three experiment sets. Each time you run the script with a different experiment name, it will add that set's data to the same results CSV file. When you are done with all three, you will have one fully populated spreadsheet.

### Customization (Optional)

You can override the default settings by setting environment variables before running the script:

```bash
# Example of running with a different port and duration
export PORT=8001
export DURATION=60
./run_and_process_sprint4.sh tensor-parallelism
```

You can also specify a different path for the source CSV file:

```bash
# Example with a custom sheet path
./run_and_process_sprint4.sh tensor-parallelism "my_custom_sprint4_sheet.csv"
```
