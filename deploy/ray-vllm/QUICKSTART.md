# Quick Start Guide

This guide will help you get Ray Serve + vLLM running on your OpenStack instance quickly.

## Step-by-Step Setup

### Step 1: Create OpenStack Instance

✅ Create an OpenStack instance with:
- Sufficient GPUs (recommended: 4-8 GPUs)
- Sufficient memory (at least 32GB, more for larger models)
- CUDA-enabled GPU drivers pre-installed (or install them manually)

### Step 2: Connect to Your Instance

```bash
ssh your-user@your-openstack-instance-ip
```

### Step 3: Get the Code on Your Instance

You have several options:

**Option A: Clone from Git Repository (Recommended)**
```bash
# If your code is in a Git repository
git clone <your-repo-url>
cd inf4RAG/deploy/ray-vllm
```

**Option B: Code Already on Instance**
```bash
# If the code is already on the instance (e.g., from a volume or previous setup)
cd /path/to/inf4RAG/deploy/ray-vllm
```

**Option C: Upload via SCP (Only if needed)**
```bash
# On your local machine
scp -r deploy/ray-vllm your-user@your-openstack-instance-ip:/path/to/destination/

# Then on the instance
cd /path/to/destination/ray-vllm
```

**Option D: Use Docker (Advanced)**
```bash
# If you prefer containerized deployment, see DEPLOYMENT.md for Docker setup
```

### Step 4: Install Dependencies

```bash
cd deploy/ray-vllm

# Install Python dependencies
pip install -r requirements.txt

# Or if you need to use pip3:
pip3 install -r requirements.txt
```

**Note:** This may take several minutes as it installs Ray, vLLM, PyTorch, etc.

### Step 5: Prepare Model Files

Ensure your model is accessible. Common locations:
- `/data/your-model-name` (if you mounted a volume)
- `~/models/your-model-name` (in your home directory)
- Any other accessible path

**If using Hugging Face models:**
```bash
# The model will be downloaded automatically on first use
# Or download manually:
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/Phi-3-mini-4k-instruct', local_dir='/data/Phi-3-mini-4k-instruct')"
```

### Step 6: Verify GPU Availability

```bash
# Check if GPUs are detected
nvidia-smi

# Or using Python
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Step 7: Start the Service

**Simplest way (auto-detect everything):**
```bash
bash launch.sh --model-path /data/your-model-path
```

**Or with Python directly:**
```bash
python3 ray_vllm_serve.py --model-path /data/your-model-path
```

The script will:
- ✅ Auto-detect available GPUs
- ✅ Auto-configure number of replicas
- ✅ Start Ray Serve + vLLM service

### Step 8: Verify Service is Running

**In another terminal (or in background):**
```bash
# Test the service
python3 test_service.py --url http://localhost:8000

# Or using curl
curl -X POST http://localhost:8000/VLLMDeployment \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 1.0,
        "max_tokens": 100
    }'
```

## Common Issues and Solutions

### Issue 1: "No GPUs detected"

**Solution:**
```bash
# Check GPU drivers
nvidia-smi

# If nvidia-smi doesn't work, install GPU drivers
# (This depends on your OpenStack instance setup)
```

### Issue 2: "Module not found: ray" or "Module not found: vllm"

**Solution:**
```bash
# Make sure dependencies are installed
pip install -r requirements.txt

# If using a virtual environment, activate it first
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 3: "Model path not found"

**Solution:**
```bash
# Check if model path exists
ls -la /data/your-model-path

# Or specify the correct path
bash launch.sh --model-path /correct/path/to/model
```

### Issue 4: "Port 8000 already in use"

**Solution:**
```bash
# Use a different port
bash launch.sh --model-path /data/your-model --port 8001

# Or find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

## Running in Background

To run the service in the background:

```bash
# Using nohup
nohup bash launch.sh --model-path /data/your-model > ray-vllm.log 2>&1 &

# Or using screen
screen -S ray-vllm
bash launch.sh --model-path /data/your-model
# Press Ctrl+A then D to detach

# Or using tmux
tmux new -s ray-vllm
bash launch.sh --model-path /data/your-model
# Press Ctrl+B then D to detach
```

## Checking Service Status

```bash
# Check if Ray is running
ray status

# Check Ray Serve status
serve status

# View logs
tail -f /tmp/ray/session_latest/logs/serve.log
```

## Stopping the Service

```bash
# If running in foreground, press Ctrl+C

# If running in background, find and kill the process
ps aux | grep ray_vllm_serve
kill <process-id>

# Or stop Ray
ray stop
```

## Next Steps

Once your service is running:

1. **Test the API** using `test_service.py` or curl
2. **Integrate with your application** by pointing to `http://your-instance-ip:8000/VLLMDeployment`
3. **Monitor performance** using Ray Dashboard (if enabled)
4. **Scale as needed** by adjusting `--num-replicas` and `--gpus-per-replica`

## Summary Checklist

Before you can use the service, make sure:

- [ ] OpenStack instance created with GPUs
- [ ] Connected to the instance via SSH
- [ ] Code uploaded/cloned to the instance
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model files accessible at specified path
- [ ] GPUs detected (`nvidia-smi` works)
- [ ] Service started successfully
- [ ] Service tested and responding

Once all checkboxes are done, your service is ready to use! 🎉

