import os
import json
import tarfile
import pandas as pd
import click
from collections import defaultdict

# --- Constants ---
METRICS_OF_INTEREST = {
    "ttft_ms": "TTFT",
    "tpot_ms": "TPOT",
    "itl_ms": "ITL"
}

STATISTICS_OF_INTEREST = {
    "mean": "Mean",
    "median": "Median",
    "p99": "P99"
}

# Column headers from the CSV for each experiment set
COLUMN_MAP = {
    "tensor-parallelism": {
        "steady": ["C", "D", "E", "F", "G"],
        "flood": ["H"]
    },
    "pipeline-parallelism": {
        "steady": ["I", "J", "K", "L", "M"],
        "flood": ["N"]
    },
    "pd-disaggregation": {
        "steady": ["O", "P", "Q", "R", "S"],
        "flood": ["T"]
    }
}

# --- Helper Functions ---

def find_archive_files(results_dir: str):
    """Find all .tar.gz files in the given directory."""
    archives = []
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".tar.gz"):
            archives.append(os.path.join(results_dir, filename))
    return archives

def extract_request_rate_from_filename(filename: str) -> float | None:
    """Extracts the request rate from the benchmark archive filename."""
    # Example: bench_..._steady_request-rate-16.0_...tar.gz
    parts = filename.split('_')
    for part in parts:
        if part.startswith("request-rate-"):
            try:
                rate_str = part.replace("request-rate-", "")
                return float(rate_str)
            except ValueError:
                return None
    return None

def parse_benchmark_result(archive_path: str) -> dict | None:
    """Extracts key metrics from the results.json inside a .tar.gz archive."""
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Find the results.json file in the tar archive
            json_file_member = None
            for member in tar.getmembers():
                if member.name.endswith("results.json"):
                    json_file_member = member
                    break
            
            if not json_file_member:
                click.secho(f"  [!] 'results.json' not found in {os.path.basename(archive_path)}", fg="yellow")
                return None

            # Extract and load the JSON data
            json_file = tar.extractfile(json_file_member)
            if not json_file:
                return None
            
            data = json.load(json_file)
            
            # Extract the summary statistics
            summary = data.get("summary", {})
            results = {}
            for key, metric_name in METRICS_OF_INTEREST.items():
                if key in summary:
                    results[metric_name] = {
                        stat_key: summary[key].get(stat_key)  # Keep lowercase keys
                        for stat_key, stat_name in STATISTICS_OF_INTEREST.items()
                    }
            return results
            
    except Exception as e:
        click.secho(f"  [!] Error processing {os.path.basename(archive_path)}: {e}", fg="red")
        return None

# --- Main Logic ---

@click.command()
@click.argument('results_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('sheet_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output-path', default="vLLM_benchmark_EC528_-_Sprint4_RESULTS.csv", help="Path to save the updated CSV.")
def process_results(results_dir, sheet_path, output_path):
    """
    Processes benchmark archives, extracts metrics, and fills them into a target CSV sheet.

    RESULTS_DIR: The directory containing the benchmark result archives (.tar.gz), 
                 e.g., 'sprint4_results/tensor-parallelism'.
    
    SHEET_PATH: The path to the original CSV file exported from Google Sheets.
    """
    # Identify the experiment set from the results directory path
    experiment_set = os.path.basename(os.path.normpath(results_dir))
    if experiment_set not in COLUMN_MAP:
        click.secho(f"ERROR: Invalid results directory. Expected one of {list(COLUMN_MAP.keys())}", fg="red")
        click.secho(f"Got: '{experiment_set}' from path '{results_dir}'", fg="red")
        return

    click.secho(f"--- Processing benchmark set: {experiment_set} ---", fg="cyan")
    
    # Load the original CSV into a pandas DataFrame
    df = pd.read_csv(sheet_path, header=None)
    
    # Find all archive files
    archive_files = find_archive_files(results_dir)
    if not archive_files:
        click.secho("No .tar.gz archives found to process.", fg="yellow")
        return

    click.echo(f"Found {len(archive_files)} archive(s) to process.")

    # Process each archive
    for archive_path in archive_files:
        filename = os.path.basename(archive_path)
        click.echo(f"\n-> Processing: {filename}")
        
        results = parse_benchmark_result(archive_path)
        if not results:
            continue

        is_flood = "flood" in filename
        
        if is_flood:
            # Handle flood test
            col_letter = COLUMN_MAP[experiment_set]["flood"][0]
            col_index = ord(col_letter) - ord('A')
            click.secho(f"   Type: Flood | Target Column: {col_letter}", fg="blue")
            
            for metric, stats in results.items():
                for stat_key, stat_name in STATISTICS_OF_INTEREST.items():
                    # Find the correct row index
                    # For Mean: column A has the metric name, column B has "Mean"
                    # For Median/P99: column A is empty, column B has "Median"/"P99"
                    row_mask = ((df.iloc[:, 0] == metric) | (df.iloc[:, 0].isna()) | (df.iloc[:, 0] == "")) & (df.iloc[:, 1] == stat_name)
                    if row_mask.any():
                        row_index = df[row_mask].index[0]
                        value = stats.get(stat_key)  # Use stat_key (lowercase) not stat_name
                        if value is not None:
                            df.iat[row_index, col_index] = round(value, 4)
                            click.echo(f"     - Wrote {metric}/{stat_name}: {value:.4f} to ({row_index+1}, {col_letter})")

        else: # Steady test
            rate = extract_request_rate_from_filename(filename)
            if rate is None:
                click.secho(f"   [!] Could not determine request rate from filename. Skipping.", fg="yellow")
                continue
            
            # Find the correct column for this request rate
            try:
                # The TPS values in the sheet are at row 3 (0-indexed)
                tps_row = df.iloc[3].values
                steady_cols = COLUMN_MAP[experiment_set]["steady"]
                
                target_col_letter = None
                for col_letter in steady_cols:
                    col_idx = ord(col_letter) - ord('A')
                    csv_value = tps_row[col_idx]
                    # Try to compare as numbers (handle both int and float representations)
                    try:
                        if float(csv_value) == float(rate):
                            target_col_letter = col_letter
                            break
                    except (ValueError, TypeError):
                        # If conversion fails, try string comparison
                        if str(csv_value) == str(int(rate)):
                            target_col_letter = col_letter
                            break
                
                if not target_col_letter:
                    click.secho(f"   [!] Could not find a column for request rate {rate} in the sheet. Skipping.", fg="yellow")
                    continue

                col_index = ord(target_col_letter) - ord('A')
                click.secho(f"   Type: Steady | Rate: {rate} req/s | Target Column: {target_col_letter}", fg="blue")

                for metric, stats in results.items():
                    for stat_key, stat_name in STATISTICS_OF_INTEREST.items():
                        # For Mean: column A has the metric name, column B has "Mean"
                        # For Median/P99: column A is empty, column B has "Median"/"P99"
                        row_mask = ((df.iloc[:, 0] == metric) | (df.iloc[:, 0].isna()) | (df.iloc[:, 0] == "")) & (df.iloc[:, 1] == stat_name)
                        if row_mask.any():
                            row_index = df[row_mask].index[0]
                            value = stats.get(stat_key)  # Use stat_key (lowercase) not stat_name
                            if value is not None:
                                df.iat[row_index, col_index] = round(value, 4)
                                click.echo(f"     - Wrote {metric}/{stat_name}: {value:.4f} to ({row_index+1}, {target_col_letter})")

            except Exception as e:
                click.secho(f"   [!] An error occurred during steady processing: {e}", fg="red")

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=False, header=False)
    click.secho(f"\n--- Successfully saved updated sheet to: {output_path} ---", fg="green")


if __name__ == "__main__":
    process_results()
