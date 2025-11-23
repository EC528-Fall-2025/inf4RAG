# Ray Cluster + vLLM Multi-Node Deployment

Automated deployment of multiple vLLM instances across multiple GPU nodes on OpenStack, coordinated via a Ray cluster.

## Overview

This solution automates the entire deployment process described in `howtodeplyray.md`:
- **Multi-Node Setup**: Deploy across multiple GPU nodes
- **Ray Cluster**: Automatic Ray cluster setup (head + workers)
- **vLLM Services**: Independent vLLM server on each node
- **Full Automation**: One command deploys everything
- **Configuration-Driven**: All settings in `config.yaml`

## Architecture

```
OpenStack Instances
│
├── Ray Head Node (ray-node-1)
│   ├── Ray Head (Port 6379)
│   ├── Ray Dashboard (Port 8265)
│   └── vLLM Server (Port 8000)
│
└── Ray Worker Nodes (ray-node-2, ...)
    ├── Ray Worker
    └── vLLM Server (Port 8000)
```

Each node runs an independent vLLM server accessible via its Floating IP.

## Prerequisites

1. **OpenStack Instances Created**:
   - At least 2 GPU instances (1 head + 1+ workers)
   - Floating IPs assigned to each instance
   - Security groups configured (SSH, Ray ports, vLLM port)

2. **Local Machine**:
   - SSH private key for accessing instances
   - Python 3.8+ with PyYAML (`pip install pyyaml`)

## Quick Start

### Step 1: Configure `config.yaml`

After creating OpenStack instances and getting Floating IPs, edit `config.yaml`:

```yaml
nodes:
  - name: "ray-node-1"
    floating_ip: "199.94.61.26"  # Your Floating IP
    role: "head"
    username: "ubuntu"
  
  - name: "ray-node-2"
    floating_ip: "199.94.61.27"  # Your Floating IP
    role: "worker"
    username: "ubuntu"

ssh:
  key_path: "~/.ssh/id_rsa"  # Your SSH private key

vllm:
  model: "gpt2"  # Your model name
```

### Step 2: Run Automated Deployment

```bash
cd deploy/ray-vllm
bash deploy.sh
```

That's it! The script will automatically:
1. ✅ Check SSH connectivity
2. ✅ Install NVIDIA drivers (if needed)
3. ✅ Install PyTorch with CUDA
4. ✅ Install Ray and vLLM
5. ✅ Start Ray cluster
6. ✅ Start vLLM services on each node
7. ✅ Test all services

## Configuration

All configuration is in `config.yaml`. Key sections:

### Node Configuration

```yaml
nodes:
  - name: "ray-node-1"
    floating_ip: "<FLOATING_IP>"  # Required: Fill in your Floating IP
    role: "head"                 # "head" or "worker"
    username: "ubuntu"           # SSH username (usually "ubuntu")
```

**Important**: Replace `<FLOATING_IP>` with your actual Floating IPs!

### SSH Configuration

```yaml
ssh:
  key_path: "~/.ssh/id_rsa"              # Path to SSH private key
  strict_host_key_checking: false        # Skip host key verification
```

### vLLM Configuration

```yaml
vllm:
  model: "gpt2"                    # HuggingFace model name or local path
  port: 8000                       # vLLM API port
  max_model_len: 512               # Maximum model length
  gpu_per_node: 1                  # GPUs per node (CUDA_VISIBLE_DEVICES)
```

### Deployment Options

```yaml
deployment:
  skip_driver_install: false      # Skip NVIDIA driver installation
  skip_ray_install: false          # Skip Ray installation
  skip_vllm_install: false         # Skip vLLM installation
  skip_ray_cluster: false          # Skip Ray cluster setup
  skip_vllm_start: false           # Skip vLLM service start
```

Use these to resume deployment after partial completion.

## Deployment Process

The `deploy.sh` script automates these steps (from `howtodeplyray.md`):

1. **SSH Connectivity Check**: Verifies access to all nodes
2. **NVIDIA Driver Installation**: Installs drivers on all nodes (if enabled)
3. **PyTorch Installation**: Installs PyTorch with CUDA support
4. **Ray Installation**: Installs Ray on all nodes
5. **vLLM Installation**: Installs vLLM on all nodes
6. **Ray Cluster Setup**: 
   - Starts Ray head on head node
   - Connects workers to head
   - Verifies cluster
7. **vLLM Services**: Starts vLLM server on each node
8. **Service Testing**: Tests all vLLM endpoints

## Usage After Deployment

### Access Ray Dashboard

```
http://<HEAD_NODE_FLOATING_IP>:8265
```

### Access vLLM APIs

Each node has its own vLLM endpoint:

```bash
# Head node
curl http://<HEAD_FLOATING_IP>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "Hello", "max_tokens": 16}'

# Worker node
curl http://<WORKER_FLOATING_IP>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "Hello", "max_tokens": 16}'
```

### Check Service Status

```bash
# Check vLLM logs on a node
ssh -i ~/.ssh/id_rsa ubuntu@<FLOATING_IP> 'tail -f /tmp/vllm.log'

# Check Ray status
ssh -i ~/.ssh/id_rsa ubuntu@<HEAD_FLOATING_IP> 'ray status'
```

## Troubleshooting

### SSH Connection Failed

- Check Floating IPs are correct in `config.yaml`
- Verify SSH key path is correct
- Ensure security group allows SSH (port 22)
- Test manually: `ssh -i ~/.ssh/id_rsa ubuntu@<FLOATING_IP>`

### Driver Installation Issues

- Check if drivers are already installed: `nvidia-smi`
- Set `skip_driver_install: true` if drivers are pre-installed
- Verify GPU is detected after reboot

### Ray Cluster Not Starting

- Check Ray head is running: `ray status` on head node
- Verify ports 6379 and 8265 are open in security groups
- Check firewall rules on instances

### vLLM Service Not Responding

- Check logs: `ssh ... 'tail -f /tmp/vllm.log'`
- Verify model is accessible (HuggingFace models auto-download)
- Check GPU memory: `nvidia-smi`
- Ensure port 8000 is open in security groups

### Resume Partial Deployment

If deployment fails partway, you can resume:

```yaml
deployment:
  skip_driver_install: true   # Skip if already installed
  skip_ray_install: true        # Skip if already installed
  skip_vllm_install: true       # Skip if already installed
```

Then run `bash deploy.sh` again.

## Configuration Reference

See `config.yaml` for all available options:

- **nodes**: Node configuration (Floating IPs, roles, usernames)
- **ssh**: SSH connection settings
- **driver**: NVIDIA driver installation settings
- **cuda**: PyTorch/CUDA configuration
- **ray**: Ray cluster settings
- **vllm**: vLLM model and service settings
- **deployment**: Skip options for resuming deployment
- **testing**: Service testing configuration

## Single-Node Deployment (Alternative)

For single-node deployment with Ray Serve (2 GPU 1 node), use:

```bash
bash launch.sh
```

This uses a different configuration format. See `config.yaml` comments for details.

## Related Documentation

- **Manual Deployment**: [howtodeplyray.md](howtodeplyray.md) - Step-by-step manual process
- **Single-Node**: Use `launch.sh` for single-node Ray Serve deployment
