import os
import copy
import requests  # Make sure this is installed
import click
import math
import subprocess
import shutil
from datetime import datetime

vllm_bench_serve_template = \
"""
vllm bench serve \\
    --dataset-name random \\
    --model {model} \\
    --num-prompts {num_prompts} \\
    --random-input-len {prefill_size} \\
    --random-output-len {max_sequence_length} \\
    --host {host} \\
    --port {port} \\
    --ignore-eos \\
    --save-result \\
    --result-dir {result_dir} \\
"""

def fetch_model_name(base_url: str) -> str:
    """Fetch the available model names from the server."""
    url = f"{base_url}/models"
    try:
        response = requests.get(url, timeout=5) # Added 5-second timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        models = response.json().get("data", [])
        if not models:
            raise ValueError("No models available from the server.")
        
        # This assumes the first model listed is the one we want to test.
        model_name = models[0]["id"]
        click.secho(f"--- Fetched model name from server: {model_name} ---", fg="green")
        return model_name
        
    except requests.exceptions.ConnectionError:
        click.secho(f"--- ERROR: Could not connect to server at {url} ---", fg="red")
        click.secho("Please ensure the vLLM server is running before starting the benchmark.", fg="red")
        exit(1) # Exit the script if we can't connect
    except Exception as e:
        click.secho(f"--- ERROR: Failed to fetch model name: {e} ---", fg="red")
        exit(1)


class BenchmarkConfig:
    def __init__(self,
                 num_prompts: int,
                 prefill_size: int,
                 max_sequence_length: int, 
                 host: str, port: int,
                 model_type: str):
        self.num_prompts = num_prompts
        self.prefill_size = prefill_size
        self.max_sequence_length = max_sequence_length 
        self.host = host
        self.port = port
        self.model_type = model_type
        
        self.result_dir = None 
        
        # Define base_url *before* using it
        self.base_url = f"http://{self.host}:{self.port}/v1"
        
        # Fetch the model name from the server automatically
        self.model = fetch_model_name(self.base_url)  # NOTE: DO NOT SET "served-model-name" WHEN LAUNCHING SERVER
        
        self.additional_args = {}

    def add_new_arguments(self, **kwargs):
        for key, value in kwargs.items():
            self.additional_args[key] = value
    
    def get_command(self):
        if self.result_dir is None:
            raise ValueError("result_dir is not set. This should be set before calling get_command.")
            
        basic_command = vllm_bench_serve_template.format(
            model=self.model,
            num_prompts=self.num_prompts,
            prefill_size=self.prefill_size,
            max_sequence_length=self.max_sequence_length, # This name is used in the template
            host=self.host,
            port=self.port,
            result_dir=self.result_dir
        )

        full_command = basic_command

        # Dynamically add the correct backend and endpoint
        if self.model_type == 'chat':
            full_command += "    --backend openai-chat \\\n"
            full_command += "    --endpoint /v1/chat/completions \\\n"
        else: # 'completion'
            full_command += "    --backend openai \\\n"
            full_command += "    --endpoint /v1/completions \\\n"

        for key, value in self.additional_args.items():
            full_command += f"    --{key.replace('_', '-')} {value} \\\n"

        full_command = full_command.rstrip(" \\\n")  # Remove trailing backslash and newline

        return full_command


# --- UPDATED HELPER FUNCTION ---
# This function contains the logic to run, archive, and clean up.
def run_and_archive(config: BenchmarkConfig, test_type: str, **kwargs):
    """
    Runs the benchmark, archives results, and cleans up.
    """
    config.add_new_arguments(**kwargs)
    
    # 1. Generate a unique result directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean up model name for use in filename
    safe_model_name = config.model.replace('/','_')
    result_dir = f"bench_{safe_model_name}_{test_type}_{timestamp}"
    config.result_dir = result_dir
    
    # 2. Get the full shell command
    command_to_run = config.get_command()
    
    click.echo(f"--- Running {test_type} benchmark... ---")
    click.echo(f"Results will be saved to: {result_dir}")
    click.echo("Command to run:")
    click.echo(command_to_run)
    click.echo("-" * 50)
    
    try:
        # 3. Execute the benchmark command
        subprocess.run(command_to_run, shell=True, check=True, text=True)
        
        click.secho(f"\n--- Benchmark complete. Archiving results... ---", fg="green")
        
        # 4. Create a .tar.gz archive
        archive_name = f"{result_dir}"
        shutil.make_archive(archive_name, 'gztar', result_dir)
        
        click.secho(f"--- Successfully created archive: {archive_name}.tar.gz ---", fg="green")
        
        # 5. Clean up the original results directory
        click.echo(f"--- Cleaning up results directory: {result_dir} ---")
        shutil.rmtree(result_dir)
        click.echo("--- Done. ---")

    except subprocess.CalledProcessError as e:
        click.secho(f"\n--- Benchmark FAILED ---", fg="red")
        click.secho(f"Return code: {e.returncode}", fg="red")
        click.secho("Benchmark directory with logs (if any) was NOT deleted.", fg="yellow")
        click.secho(f"Please check directory: {result_dir}", fg="yellow")
    except Exception as e:
        click.secho(f"\n--- An unexpected error occurred ---", fg="red")
        click.secho(f"Error: {e}", fg="red")
        click.secho(f"Please check directory (if it exists): {result_dir}", fg="yellow")



@click.group()
@click.option("--num-prompts", type=int, default=1024, help="Number of prompts to generate")
@click.option("--prefill-size", type=int, default=32, help="Size of prefill in tokens")
@click.option("--max-sequence-length", type=int, default=512, help="Max tokens to generate") 
@click.option("--host", type=str, default="127.0.0.1", help="Host IP of the model server")
@click.option("--port", type=int, default=8000, help="Port of the model server")
@click.option("--model-type", type=click.Choice(['chat', 'completion']), default='chat', help="Use 'chat' for Instruct models, 'completion' for Base models.")
@click.pass_context
def bench(ctx, num_prompts, prefill_size, max_sequence_length, host, port, model_type): # <-- CHANGED
    ctx.obj = BenchmarkConfig(
        num_prompts,
        prefill_size,
        max_sequence_length,
        host,
        port,
        model_type
    )

@bench.command("steady")
@click.option("--duration", type=int, default=90, help="Duration of the steady benchmark in seconds")
@click.option("--request-rate", type=float, default=16.0, help="Requests per second")
@click.pass_obj
def steady_testing(config: BenchmarkConfig, **kwargs):
    inner_config = copy.deepcopy(config)

    # Overwrite the number of prompts
    duration = kwargs.pop("duration")
    inner_config.num_prompts = math.ceil(duration * kwargs["request_rate"])

    run_and_archive(inner_config, "steady", **kwargs)

@bench.command("flood")
@click.pass_obj
def flood_testing(config: BenchmarkConfig, **kwargs):
    inner_config = copy.deepcopy(config)
    run_and_archive(inner_config, "flood", **kwargs)

if __name__=="__main__":
    bench()

# python bench.py \
#   --model-type chat \
#   flood

# python bench.py \
#   --model-type chat \
#   steady --request-rate 32 --duration 120
