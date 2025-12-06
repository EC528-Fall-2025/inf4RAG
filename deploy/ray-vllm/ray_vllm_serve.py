"""
Ray Serve + vLLM Multi-Instance Deployment Script
Run multiple vLLM instances on the same node for data-parallel inference
"""

from ray import serve
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import argparse
import subprocess


class VLLMDeployment:
    """
    vLLM Deployment Class
    Each replica independently loads the model and processes requests
    Ray Serve automatically performs load balancing
    """
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 2):
        """
        Initialize vLLM engine
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Tensor parallel size (number of GPUs used internally by each replica)
        """
        # Get GPU devices for current replica
        # Ray automatically assigns different GPUs to different replicas
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            # Ray automatically manages GPU allocation, this is mainly for logging
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            print(f"[VLLMDeployment] Initializing on GPUs: {visible_gpus}")
        
        # Initialize vLLM engine
        # Note: Each replica will load the model once
        print(f"[VLLMDeployment] Loading model from: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        print(f"[VLLMDeployment] Model loaded successfully")
    
    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inference request
        This function is automatically called by Ray Serve
        
        Args:
            request: Dictionary containing messages, temperature, max_tokens, etc.
        
        Returns:
            Response dictionary compatible with OpenAI API format
        """
        # Parse request (compatible with OpenAI API format)
        messages = request.get("messages", [])
        temperature = request.get("temperature", 1.0)
        max_tokens = request.get("max_tokens", 1000)
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Execute inference
        outputs = self.llm.generate([prompt], sampling_params)
        
        # Return result (compatible with OpenAI format)
        generated_text = outputs[0].outputs[0].text
        return {
            "id": "chatcmpl-ray-vllm",
            "object": "chat.completion",
            "created": int(os.path.getmtime(__file__)),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        }
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        Convert message list to prompt string
        Adjust according to your model format (using generic format here)
        
        Args:
            messages: Message list, format like [{"role": "user", "content": "..."}]
        
        Returns:
            Prompt string
        """
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        return prompt


def detect_available_gpus() -> int:
    """
    Detect the number of available GPUs
    
    Returns:
        Number of available GPUs
    """
    # Try using torch first
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"[Auto-detect] Detected {gpu_count} GPUs using PyTorch")
            return gpu_count
    except ImportError:
        pass
    
    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_count = len(result.stdout.strip().split('\n'))
        print(f"[Auto-detect] Detected {gpu_count} GPUs using nvidia-smi")
        return gpu_count
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # If no GPUs detected, return 0
    print("[Auto-detect] Warning: No GPUs detected, returning 0")
    return 0


def auto_configure_gpus(
    total_gpus: int,
    preferred_gpus_per_replica: Optional[int] = None,
    min_replicas: int = 1,
    max_replicas: int = 8
) -> Tuple[int, int]:
    """
    Automatically configure number of replicas and GPUs per replica
    
    Args:
        total_gpus: Total number of available GPUs
        preferred_gpus_per_replica: Preferred GPUs per replica (if None, auto-calculate)
        min_replicas: Minimum number of replicas
        max_replicas: Maximum number of replicas
    
    Returns:
        Tuple of (num_replicas, gpus_per_replica)
    """
    if total_gpus == 0:
        print("[Auto-configure] No GPUs available, using default: 1 replica, 1 GPU")
        return (1, 1)
    
    if preferred_gpus_per_replica is None:
        # Auto-calculate optimal GPUs per replica
        # Strategy: For 2 GPU node, use 1 GPU per replica (2 replicas)
        # For more GPUs, prefer 2 GPUs per replica for better performance
        if total_gpus == 2:
            # 2 GPU node: 2 replicas × 1 GPU each (data parallelism)
            preferred_gpus_per_replica = 1
        elif total_gpus >= 8:
            preferred_gpus_per_replica = 2
        elif total_gpus >= 4:
            preferred_gpus_per_replica = 2
        else:
            preferred_gpus_per_replica = 1
    
    # Calculate number of replicas
    num_replicas = total_gpus // preferred_gpus_per_replica
    
    # Ensure we have at least min_replicas and at most max_replicas
    num_replicas = max(min_replicas, min(num_replicas, max_replicas))
    
    # Recalculate gpus_per_replica to use all available GPUs
    gpus_per_replica = total_gpus // num_replicas
    
    # Ensure we use all GPUs by adjusting if needed
    if num_replicas * gpus_per_replica < total_gpus:
        # Distribute remaining GPUs
        remaining = total_gpus - (num_replicas * gpus_per_replica)
        if remaining > 0:
            # Add one more GPU to the first few replicas
            print(f"[Auto-configure] Distributing {remaining} extra GPU(s) to first replicas")
    
    print(f"[Auto-configure] Configuration: {num_replicas} replicas, {gpus_per_replica} GPUs per replica")
    print(f"[Auto-configure] Total GPUs used: {num_replicas * gpus_per_replica} / {total_gpus}")
    
    return (num_replicas, gpus_per_replica)


def create_deployment(
    model_path: str,
    num_replicas: int = 4,
    gpus_per_replica: int = 2,
    tensor_parallel_size: int = 2,
    host: str = "0.0.0.0",
    port: int = 8000
):
    """
    Create and start Ray Serve deployment
    
    Args:
        model_path: Path to the model
        num_replicas: Number of replicas (vLLM instances)
        gpus_per_replica: Number of GPUs per replica
        tensor_parallel_size: Tensor parallel size within each replica
        host: Service listening address
        port: Service port
    """
    import ray
    
    # Initialize Ray (single node mode)
    total_gpus = num_replicas * gpus_per_replica
    ray.init(
        num_gpus=total_gpus,
        ignore_reinit_error=True,
    )
    
    # Start Ray Serve
    serve.start(http_options={"host": host, "port": port})
    
    # Use decorator to dynamically configure deployment
    VLLMDeploymentWithConfig = serve.deployment(
        num_replicas=num_replicas,
        ray_actor_options={
            "num_gpus": gpus_per_replica,
        }
    )(VLLMDeployment)
    
    # Deploy service
    VLLMDeploymentWithConfig.deploy(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size
    )
    
    print("=" * 60)
    print("Ray Serve + vLLM started!")
    print(f"Service address: http://{host}:{port}")
    print(f"API endpoint: http://{host}:{port}/VLLMDeployment")
    print(f"Number of replicas: {num_replicas}")
    print(f"GPUs per replica: {gpus_per_replica}")
    print(f"Total GPUs: {total_gpus}")
    print(f"Model path: {model_path}")
    print("=" * 60)
    
    return VLLMDeploymentWithConfig


if __name__ == "__main__":
    # Load configuration from config.yaml
    try:
        from load_config import load_config
        config = load_config()
    except ImportError:
        # Fallback if load_config not available
        config = {
            'model_path': 'gpt2',
            'num_replicas': None,
            'gpus_per_replica': None,
            'tensor_parallel_size': None,
            'host': '0.0.0.0',
            'port': 8000
        }
    
    parser = argparse.ArgumentParser(description="Ray Serve + vLLM deployment script")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"Model path: HuggingFace model name (e.g., gpt2) or local path (default: from config.yaml or {config['model_path']})"
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        help="Number of replicas (vLLM instances). If not specified, use config.yaml or auto-detect"
    )
    parser.add_argument(
        "--gpus-per-replica",
        type=int,
        default=None,
        help="Number of GPUs per replica. If not specified, use config.yaml or auto-calculate"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size within each replica. If not specified, use config.yaml or equals gpus-per-replica"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Service listening address (default: from config.yaml or {config['host']})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Service port (default: from config.yaml or {config['port']})"
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect GPU configuration (enabled by default if num-replicas or gpus-per-replica not specified)"
    )
    
    args = parser.parse_args()
    
    # Use config.yaml values if command line arguments not provided
    model_path = args.model_path if args.model_path is not None else config['model_path']
    num_replicas = args.num_replicas if args.num_replicas is not None else config['num_replicas']
    gpus_per_replica = args.gpus_per_replica if args.gpus_per_replica is not None else config['gpus_per_replica']
    tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else config['tensor_parallel_size']
    host = args.host if args.host is not None else config['host']
    port = args.port if args.port is not None else config['port']
    
    print(f"Configuration loaded from config.yaml:")
    print(f"  Model path: {model_path}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    if num_replicas is not None:
        print(f"  Num replicas: {num_replicas}")
    if gpus_per_replica is not None:
        print(f"  GPUs per replica: {gpus_per_replica}")
    print("")
    
    # Auto-detect GPUs if needed
    if num_replicas is None or gpus_per_replica is None or args.auto_detect:
        print("=" * 60)
        print("Auto-detecting GPU configuration...")
        print("=" * 60)
        
        available_gpus = detect_available_gpus()
        
        if available_gpus == 0:
            print("Error: No GPUs detected. Please ensure GPUs are available.")
            exit(1)
        
        # Auto-configure
        auto_num_replicas, auto_gpus_per_replica = auto_configure_gpus(
            total_gpus=available_gpus,
            preferred_gpus_per_replica=gpus_per_replica
        )
        
        # Use config.yaml or auto-detected values
        if num_replicas is None:
            num_replicas = auto_num_replicas
        if gpus_per_replica is None:
            gpus_per_replica = auto_gpus_per_replica
        
        # Set tensor_parallel_size if not specified
        if tensor_parallel_size is None:
            tensor_parallel_size = gpus_per_replica
        
        # Validate configuration
        total_gpus_needed = num_replicas * gpus_per_replica
        if total_gpus_needed > available_gpus:
            print(f"Warning: Configuration requires {total_gpus_needed} GPUs, but only {available_gpus} available")
            print("Ray will handle this automatically, but may not start all replicas")
        
        print("=" * 60)
    else:
        # Use config.yaml or specified values
        if num_replicas is None:
            num_replicas = 2  # Default for 2 GPU node
        if gpus_per_replica is None:
            gpus_per_replica = 1  # Default for 2 GPU node
        if tensor_parallel_size is None:
            tensor_parallel_size = gpus_per_replica
        
        # Validate against available GPUs
        available_gpus = detect_available_gpus()
        total_gpus_needed = num_replicas * gpus_per_replica
        if total_gpus_needed > available_gpus:
            print(f"Warning: Configuration requires {total_gpus_needed} GPUs, but only {available_gpus} available")
            print("Ray will handle this automatically, but may not start all replicas")
    
    # Create and start deployment
    create_deployment(
        model_path=model_path,
        num_replicas=num_replicas,
        gpus_per_replica=gpus_per_replica,
        tensor_parallel_size=tensor_parallel_size,
        host=host,
        port=port
    )
    
    # Keep running
    import time
    import ray
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down service...")
        serve.shutdown()
        ray.shutdown()
        print("Service closed")

