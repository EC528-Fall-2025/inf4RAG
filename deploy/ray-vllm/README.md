# Ray Serve + vLLM Multi-Instance Deployment

This directory contains a deployment solution for running multiple vLLM instances on the same node using Ray Serve.

## Overview

Ray Serve + vLLM allows you to run multiple vLLM instances on a single node, enabling:
- **Data Parallelism**: Each instance processes independent request batches
- **Automatic Load Balancing**: Ray Serve automatically distributes requests to different instances
- **Dynamic Scaling**: Adjust the number of instances at any time
- **High Availability**: Automatic restart when instances crash

## Architecture

```
OpenStack Instance (e.g., 8 GPU node)
│
├── Ray Head Node (Management Node)
│   └── Ray Serve API (Port 8000)
│
└── Ray Worker Nodes (Worker nodes, can be the same machine)
    ├── vLLM Instance 1 (GPU 0-1)
    ├── vLLM Instance 2 (GPU 2-3)
    ├── vLLM Instance 3 (GPU 4-5)
    └── vLLM Instance 4 (GPU 6-7)
```

## Prerequisites

1. **OpenStack Instance**: Need sufficient GPUs and memory
2. **Python 3.8+**
3. **CUDA and cuDNN**: For GPU acceleration
4. **Model Files**: Ensure model path is accessible

> 📖 **New to this?** Check out [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide!
> 
> 🚀 **Deployment Options:** See [DEPLOYMENT_OPTIONS.md](DEPLOYMENT_OPTIONS.md) for different ways to get code on your instance (Git, Docker, etc.)

## Installation

### 1. Install Dependencies

```bash
cd deploy/ray-vllm
pip install -r requirements.txt
```

### 2. Configure Model Path

Edit `config.yaml` or use environment variables/command line arguments to specify the model path:

```bash
export MODEL_PATH=/data/Phi-3-mini-4k-instruct
```

## Usage

### Method 1: Using Launch Script (Recommended)

```bash
# Auto-detect GPU configuration (recommended)
bash launch.sh --model-path /data/your-model

# Custom configuration (override auto-detection)
bash launch.sh \
    --model-path /data/your-model \
    --num-replicas 4 \
    --gpus-per-replica 2 \
    --port 8000

# Disable auto-detection
bash launch.sh --model-path /data/your-model --no-auto-detect --num-replicas 2
```

### Method 2: Run Python Script Directly

```bash
# Auto-detect GPU configuration (recommended)
python3 ray_vllm_serve.py --model-path /data/Phi-3-mini-4k-instruct

# Custom configuration
python3 ray_vllm_serve.py \
    --model-path /data/Phi-3-mini-4k-instruct \
    --num-replicas 4 \
    --gpus-per-replica 2 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

### Parameter Description

- `--model-path`: Model path (required)
- `--num-replicas`: Number of replicas, i.e., vLLM instances (auto-detect if not specified)
- `--gpus-per-replica`: Number of GPUs per replica (auto-detect if not specified)
- `--tensor-parallel-size`: Tensor parallel size within each replica (auto-detect if not specified, equals gpus-per-replica)
- `--host`: Service listening address (default: 0.0.0.0)
- `--port`: Service port (default: 8000)
- `--auto-detect`: Enable auto-detection of GPU configuration (enabled by default if num-replicas or gpus-per-replica not specified)

### Auto-Detection Feature

The script automatically detects available GPUs and configures the deployment:

- **GPU Detection**: Uses PyTorch (`torch.cuda.device_count()`) or `nvidia-smi` to detect available GPUs
- **Auto-Configuration**: Automatically calculates optimal number of replicas and GPUs per replica
- **Strategy**: Prefers 2 GPUs per replica for better performance, distributes remaining GPUs evenly

**Examples of auto-configuration:**
- 8 GPUs → 4 replicas × 2 GPUs each
- 4 GPUs → 2 replicas × 2 GPUs each
- 2 GPUs → 1 replica × 2 GPUs
- 1 GPU → 1 replica × 1 GPU

## Configuration Examples

### Example 1: Auto-Detection (Recommended)

```bash
# Automatically detect and configure based on available GPUs
bash launch.sh --model-path /data/Phi-3-mini-4k-instruct
```

### Example 2: 8 GPU Node, 4 Instances (Manual)

```bash
# Each instance uses 2 GPUs, 4 instances total
bash launch.sh \
    --model-path /data/Phi-3-mini-4k-instruct \
    --num-replicas 4 \
    --gpus-per-replica 2
```

### Example 3: 8 GPU Node, 2 Instances (Manual)

```bash
# Each instance uses 4 GPUs, 2 instances total
bash launch.sh \
    --model-path /data/Phi-3-mini-4k-instruct \
    --num-replicas 2 \
    --gpus-per-replica 4
```

### Example 4: 4 GPU Node, 2 Instances (Manual)

```bash
# Each instance uses 2 GPUs, 2 instances total
bash launch.sh \
    --model-path /data/Phi-3-mini-4k-instruct \
    --num-replicas 2 \
    --gpus-per-replica 2
```

## Testing the Service

### 1. Check Service Status

```bash
# View Ray status
ray status

# View Ray Serve status
serve status
```

### 2. Use Test Script (Recommended)

```bash
# Test local service
python3 test_service.py

# Test remote service
python3 test_service.py --url http://YOUR_OPENSTACK_IP:8000

# Use custom prompt
python3 test_service.py --url http://YOUR_OPENSTACK_IP:8000 --prompt "What is AI?"
```

### 3. Manual API Testing

Using Python:

```python
import requests
import json

url = "http://YOUR_OPENSTACK_IP:8000/VLLMDeployment"

payload = {
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 1.0,
    "max_tokens": 100
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

Or using curl:

```bash
curl -X POST http://YOUR_OPENSTACK_IP:8000/VLLMDeployment \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 1.0,
        "max_tokens": 100
    }'
```

## Monitoring and Management

### View Deployment Status

```python
import ray
ray.init()
serve.status()  # View status of all deployments
```

### Dynamic Scaling

```python
# Scale to 6 replicas
deployment = serve.get_deployment("VLLMDeployment")
deployment.set_num_replicas(6)
```

### View Logs

```bash
# Ray log location
tail -f /tmp/ray/session_latest/logs/serve.log
```

## Notes

1. **GPU Resources**: Ensure total GPUs >= `num_replicas * gpus_per_replica`
2. **Memory Usage**: Each replica loads the model once, ensure sufficient memory
3. **Port Conflicts**: Ensure the specified port is not in use
4. **Model Path**: Ensure the model path is accessible on the OpenStack instance
5. **Tensor Parallelism**: `tensor_parallel_size` should equal `gpus_per_replica`

## Troubleshooting

### Issue 1: Insufficient GPUs

**Symptoms**: Some replicas cannot start

**Solution**:
- Reduce `num_replicas` or `gpus_per_replica`
- Check `nvidia-smi` to confirm available GPU count

### Issue 2: Insufficient Memory

**Symptoms**: Model loading fails or OOM

**Solution**:
- Reduce number of replicas
- Use a smaller model
- Increase system memory

### Issue 3: Port Already in Use

**Symptoms**: Startup fails, port already in use

**Solution**:
- Use `--port` parameter to specify another port
- Check and close processes using the port

### Issue 4: Incorrect Model Path

**Symptoms**: Cannot load model

**Solution**:
- Confirm model path is correct
- Check file permissions
- Ensure model files are complete

## Integration with Existing Systems

### Integration with WebChat

Modify `chatbotbasic/WebChat/config.yaml`:

```yaml
base_url: "http://YOUR_OPENSTACK_IP:8000/VLLMDeployment"
```

Note: Ray Serve's API format is slightly different from the standard OpenAI API, client code may need adjustment.

## Performance Optimization Recommendations

1. **Number of Replicas**: Adjust based on GPU count and load conditions
2. **Batching**: Ray Serve automatically batches requests
3. **Monitoring**: Use Ray Dashboard to monitor performance
4. **Caching**: Consider adding a response caching layer

## Related Resources

- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Ray Serve + vLLM Example](https://docs.ray.io/en/latest/serve/tutorials/vllm.html)

## Support

If you encounter issues, please check:
1. Ray and vLLM logs
2. GPU and memory usage
3. Network connection and port status
