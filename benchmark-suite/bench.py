import os
import copy
import requests  # Make sure this is installed
import click
import subprocess
import shutil
from datetime import datetime

vllm_bench_serve_template = \
"""
vllm bench serve \\
    --dataset-name random \\
    --model {tokenizer_path} \\
    --served-model-name {served_model_name} \\
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
                 random_output_len: int, 
                 host: str, port: int,
                 tokenizer_path: str,
                 model_type: str):
        self.num_prompts = num_prompts
        self.prefill_size = prefill_size
        self.max_sequence_length = random_output_len 
        self.host = host
        self.port = port
        self.tokenizer_path = tokenizer_path
        self.model_type = model_type
        
        self.result_dir = None 
        
        # Define base_url *before* using it
        self.base_url = f"http://{self.host}:{self.port}/v1"
        
        # Fetch the model name from the server automatically
        self.served_model_name = fetch_model_name(self.base_url)
        
        self.additional_args = {}

    def add_new_arguments(self, **kwargs):
        for key, value in kwargs.items():
            self.additional_args[key] = value
    
    def get_command(self):
        if self.result_dir is None:
            raise ValueError("result_dir is not set. This should be set before calling get_command.")
            
        basic_command = vllm_bench_serve_template.format(
            tokenizer_path=self.tokenizer_path,
            served_model_name=self.served_model_name,
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
    safe_model_name = config.served_model_name.replace('/','_')
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
@click.option("--prefill-size", type=int, default=32, help="Size of prefill in tokens (vLLM arg: --random-input-len)")
@click.option("--random-output-len", type=int, default=512, help="Max tokens to generate (vLLM arg: --random-output-len)") 
@click.option("--host", type=str, default="127.0.0.1", help="Host IP of the model server")
@click.option("--port", type=int, default=8000, help="Port of the model server")
@click.option("--tokenizer-path", type=str, required=True, help="Local path to the model for tokenizer loading (e.g., /mnt/models/llama-30b)")
@click.option("--model-type", type=click.Choice(['chat', 'completion']), default='chat', help="Use 'chat' for Instruct models, 'completion' for Base models.")
@click.pass_context
def bench(ctx, num_prompts, prefill_size, random_output_len, host, port, tokenizer_path, model_type): # <-- CHANGED
    ctx.obj = BenchmarkConfig(
        num_prompts, prefill_size, random_output_len,  # <-- CHANGED
        host, port, tokenizer_path, model_type
    )

@bench.command("steady")
@click.option("--request-rate", type=float, default=16.0, help="Requests per second (vLLM arg: --request-rate)")
@click.pass_obj
def steady_testing(config: BenchmarkConfig, **kwargs):
    inner_config = copy.deepcopy(config)
    run_and_archive(inner_config, "steady", **kwargs)

@bench.command("flood")
@click.pass_obj
def flood_testing(config: BenchmarkConfig, **kwargs):
    inner_config = copy.deepcopy(config)
    run_and_archive(inner_config, "flood", **kwargs)

if __name__=="__main__":
    bench()



# How to run this script:
# Example for Llama-30B (a 'completion' model)

# python bench.py \
#   --tokenizer-path /mnt/models/Qwen3-30B-A3B-Instruct-2507 \
#   --model-type chat \
#   flood

# python bench.py \
#   --tokenizer-path /mnt/models/Qwen3-30B-A3B-Instruct-2507 \
#   --model-type chat \
#   steady --request-rate 10.0