# Deployment Options

You don't have to manually upload code to OpenStack. Here are different ways to deploy:

## Option 1: Git Clone (Recommended)

If your code is in a Git repository:

```bash
# SSH to your OpenStack instance
ssh your-user@your-instance-ip

# Clone the repository
git clone <your-repo-url>
cd inf4RAG/deploy/ray-vllm

# Install and run
pip install -r requirements.txt
bash launch.sh --model-path /data/your-model
```

**Pros:**
- ✅ Easy version control
- ✅ Easy updates (just `git pull`)
- ✅ No manual file transfer needed

**Cons:**
- Requires Git repository access
- Need internet connection on instance

## Option 2: Code Already on Instance

If the code is already available on the instance (e.g., from a mounted volume, pre-installed, or from a previous session):

```bash
# SSH to your OpenStack instance
ssh your-user@your-instance-ip

# Navigate to existing code location
cd /path/to/inf4RAG/deploy/ray-vllm

# Install and run
pip install -r requirements.txt
bash launch.sh --model-path /data/your-model
```

**Pros:**
- ✅ No transfer needed
- ✅ Fastest setup

**Cons:**
- Code must already be on the instance

## Option 3: Docker Container (Advanced)

Containerize the deployment for easier management:

### Create Dockerfile

```dockerfile
# Dockerfile for Ray Serve + vLLM
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy code
COPY deploy/ray-vllm /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Expose port
EXPOSE 8000

# Default command
CMD ["python3", "ray_vllm_serve.py", "--model-path", "/models/your-model"]
```

### Build and Run

```bash
# Build image
docker build -t ray-vllm:latest .

# Run container
docker run -d \
    --gpus all \
    -p 8000:8000 \
    -v /data/models:/models \
    --name ray-vllm \
    ray-vllm:latest
```

**Pros:**
- ✅ Consistent environment
- ✅ Easy to deploy anywhere
- ✅ Isolated dependencies

**Cons:**
- More complex setup
- Requires Docker knowledge

## Option 4: Direct Installation (Minimal Files)

If you only want to install the essential files:

```bash
# On your local machine, create a minimal package
tar -czf ray-vllm-minimal.tar.gz \
    ray_vllm_serve.py \
    launch.sh \
    requirements.txt \
    test_service.py

# Upload only essential files
scp ray-vllm-minimal.tar.gz your-user@your-instance-ip:/tmp/

# On the instance
cd /tmp
tar -xzf ray-vllm-minimal.tar.gz
pip install -r requirements.txt
python3 ray_vllm_serve.py --model-path /data/your-model
```

**Pros:**
- ✅ Minimal file transfer
- ✅ Only essential files

**Cons:**
- Missing documentation and config files
- Less convenient for updates

## Option 5: Use OpenStack Volume/Image

If you have access to OpenStack volumes or custom images:

1. **Create a custom image** with code pre-installed
2. **Mount a volume** containing the code
3. **Use object storage** (Swift) to store code

```bash
# If using a mounted volume
cd /mnt/volume-name/inf4RAG/deploy/ray-vllm

# If using object storage
swift download container-name ray-vllm.tar.gz
tar -xzf ray-vllm.tar.gz
cd ray-vllm
```

**Pros:**
- ✅ Persistent across instance recreations
- ✅ Can be shared across instances

**Cons:**
- Requires OpenStack volume/image setup
- More initial configuration

## Recommendation

**For most users:** Use **Option 1 (Git Clone)** if you have a repository, or **Option 2** if code is already on the instance.

**For production:** Consider **Option 3 (Docker)** for better isolation and reproducibility.

## Quick Comparison

| Option | Setup Time | Maintenance | Best For |
|--------|-----------|-------------|----------|
| Git Clone | ⭐⭐⭐ Fast | ⭐⭐⭐ Easy | Development, Updates |
| Already on Instance | ⭐⭐⭐ Fastest | ⭐⭐ Medium | Quick testing |
| Docker | ⭐⭐ Medium | ⭐⭐⭐ Easy | Production, Isolation |
| Minimal Files | ⭐⭐ Medium | ⭐ Hard | One-time deployment |
| Volume/Image | ⭐ Slow | ⭐⭐⭐ Easy | Persistent deployments |

## Next Steps

After getting the code on your instance, continue with:
- [Install dependencies](QUICKSTART.md#step-4-install-dependencies)
- [Prepare model files](QUICKSTART.md#step-5-prepare-model-files)
- [Start the service](QUICKSTART.md#step-7-start-the-service)

