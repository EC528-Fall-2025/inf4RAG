# 🚀 Ray Cluster Setup Guide

# Multi-Node vLLM + Ray Cluster Setup on OpenStack (GPU Edition)

This document describes the full workflow for deploying **multiple vLLM instances** across **multiple GPU nodes** on OpenStack, coordinated via a **Ray cluster**.

---

# 1. Create GPU Instances on OpenStack

## 1.1 Choose Instance Configuration
For each GPU node:

- Boot Source: Image  
- Image: Ubuntu 22.04  
- Flavor: `gpu-su-a100.1` or `gpu-su-a100.2`  
- Volume size: 150–200 GB  
- Assign a Floating IP  
- Security Groups (Ingress Rules):
  - TCP 22 (SSH)
  - TCP 6379 (Ray Head)
  - TCP 8265 (Ray Dashboard)
  - TCP 8000 (vLLM API)
  - ICMP allowed

Create at least 2 GPU nodes:

- `ray-node-1` → Ray head + GPU  
- `ray-node-2` → Ray worker + GPU  

---

# 2. SSH Access

Use your private key, not `.pub`:

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<FLOATING-IP>
```

If host key changed:

```bash
ssh-keygen -R <FLOATING-IP>
```

---

# 3. Install NVIDIA Drivers on Each GPU Node

Ubuntu images do NOT include GPU drivers. Install manually.

```bash
sudo apt update
sudo apt upgrade -y
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install -y nvidia-driver-525
sudo reboot
```

Verify GPU:

```bash
nvidia-smi
```

Expected: shows A100 GPU.

---

# 4. Verify CUDA via PyTorch

Install PyTorch GPU build:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Test:

```bash
python3 - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Num GPUs:", torch.cuda.device_count())
EOF
```

Expected:

```
CUDA available: True
Num GPUs: 1
```

---

# 5. Install Ray and vLLM on Each Node

```bash
sudo apt install -y python3-pip git
pip3 install --upgrade pip
pip3 install "ray[default]"
pip3 install vllm
```

---

# 6. Start Ray Cluster

## 6.1 Start Ray Head (ray-node-1)

Find private IP:

```bash
hostname -I
```

Example: `192.168.0.228`

Start Ray head:

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

---

## 6.2 Start Ray Worker (ray-node-2)

Connect worker to head:

```bash
ray start --address='192.168.0.228:6379'
```

---

## 6.3 Verify Ray Cluster

On ray-node-1:

```bash
python3 - << 'EOF'
import ray
ray.init(address="auto")
print("Cluster resources:", ray.cluster_resources())
EOF
```

Expected output lists both nodes (CPU + GPU resources).

---

# 7. Start vLLM on Each GPU Node

## 7.1 On ray-node-1

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model gpt2 \
  --port 8000 \
  --max-model-len 512 &
```

## 7.2 On ray-node-2

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model gpt2 \
  --port 8000 \
  --max-model-len 512 &
```

Expected logs:

```
Application startup complete.
```

---

# 8. Test Each vLLM Instance

## Test ray-node-1:

```bash
curl http://<FLOATING-IP-1>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt2",
        "prompt": "Hello from node 1",
        "max_tokens": 16
      }'
```

## Test ray-node-2:

```bash
curl http://<FLOATING-IP-2>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt2",
        "prompt": "Hello from node 2",
        "max_tokens": 16
      }'
```

Expected response:

```json
{
  "id": "...",
  "object": "text_completion",
  "choices": [
    { "text": "..." }
  ]
}
```

API logs will show:

```
POST /v1/completions ... 200 OK
Avg generation throughput: ...
GPU KV cache usage: ...
```

Both nodes must independently return valid completions.

---

# 9. Final Architecture

You now have:

- A Ray head node  
- A Ray worker node  
- Two independent GPU-backed vLLM servers  
- Two OpenAI-compatible inference endpoints  
- Fully operational distributed inference setup  

---