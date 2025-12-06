#!/bin/bash

# Automated Multi-Node Ray Cluster + vLLM Deployment Script
# This script automates the entire deployment process described in howtodeplyray.md
# Automation starts from SSH connectivity check (Step 2 in the guide)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load configuration
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    echo "Please create config.yaml with your node configuration"
    exit 1
fi

echo -e "${GREEN}Loading configuration from $CONFIG_FILE...${NC}"

# Parse YAML config using Python
eval "$(python3 << 'PYTHON_EOF'
import yaml
import sys
import os
import json

config_file = os.environ.get('CONFIG_FILE', 'config.yaml')

try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Read user configuration variables from top of file
    floating_ip_1 = config.get('FLOATING_IP_1', '<FLOATING_IP_1>')
    floating_ip_2 = config.get('FLOATING_IP_2', '<FLOATING_IP_2>')
    ssh_key_path = config.get('SSH_KEY_PATH', '~/.ssh/id_rsa')
    vllm_model = config.get('VLLM_MODEL', 'gpt2')
    
    # Replace placeholders in nodes configuration
    nodes = config.get('nodes', [])
    for node in nodes:
        if node.get('floating_ip') == '<FLOATING_IP_1>':
            node['floating_ip'] = floating_ip_1
        elif node.get('floating_ip') == '<FLOATING_IP_2>':
            node['floating_ip'] = floating_ip_2
    
    # Replace placeholders in ssh and vllm config
    ssh_key = config.get('ssh', {}).get('key_path', ssh_key_path)
    if ssh_key == '<SSH_KEY_PATH>':
        ssh_key = ssh_key_path
    ssh_key = os.path.expanduser(ssh_key)
    
    # Replace vllm model if placeholder
    vllm_model_config = config.get('vllm', {}).get('model', vllm_model)
    if vllm_model_config == '<VLLM_MODEL>':
        vllm_model_config = vllm_model
    else:
        vllm_model = vllm_model_config
    ssh_strict = config.get('ssh', {}).get('strict_host_key_checking', False)
    driver_install = config.get('driver', {}).get('install', True)
    driver_version = config.get('driver', {}).get('version', '525')
    driver_reboot = config.get('driver', {}).get('reboot_after_install', True)
    pytorch_index = config.get('cuda', {}).get('pytorch_index_url', 'https://download.pytorch.org/whl/cu121')
    ray_head_port = config.get('ray', {}).get('head_port', 6379)
    ray_dashboard_port = config.get('ray', {}).get('dashboard_port', 8265)
    ray_dashboard_host = config.get('ray', {}).get('dashboard_host', '0.0.0.0')
    # vllm_model already set above from user config
    vllm_port = config.get('vllm', {}).get('port', 8000)
    vllm_max_len = config.get('vllm', {}).get('max_model_len', 512)
    vllm_gpu_per_node = config.get('vllm', {}).get('gpu_per_node', 1)
    skip_driver = config.get('deployment', {}).get('skip_driver_install', False)
    skip_ray_install = config.get('deployment', {}).get('skip_ray_install', False)
    skip_vllm_install = config.get('deployment', {}).get('skip_vllm_install', False)
    skip_ray_cluster = config.get('deployment', {}).get('skip_ray_cluster', False)
    skip_vllm_start = config.get('deployment', {}).get('skip_vllm_start', False)
    test_services = config.get('testing', {}).get('test_services', True)
    test_timeout = config.get('testing', {}).get('test_timeout', 60)
    
    # Export as shell variables
    print(f"export SSH_KEY='{ssh_key}'")
    print(f"export SSH_STRICT={str(ssh_strict).lower()}")
    print(f"export DRIVER_INSTALL={str(driver_install).lower()}")
    print(f"export DRIVER_VERSION='{driver_version}'")
    print(f"export DRIVER_REBOOT={str(driver_reboot).lower()}")
    print(f"export PYTORCH_INDEX='{pytorch_index}'")
    print(f"export RAY_HEAD_PORT={ray_head_port}")
    print(f"export RAY_DASHBOARD_PORT={ray_dashboard_port}")
    print(f"export RAY_DASHBOARD_HOST='{ray_dashboard_host}'")
    print(f"export VLLM_MODEL='{vllm_model}'")
    print(f"export VLLM_PORT={vllm_port}")
    print(f"export VLLM_MAX_LEN={vllm_max_len}")
    print(f"export VLLM_GPU_PER_NODE={vllm_gpu_per_node}")
    print(f"export SKIP_DRIVER={str(skip_driver).lower()}")
    print(f"export SKIP_RAY_INSTALL={str(skip_ray_install).lower()}")
    print(f"export SKIP_VLLM_INSTALL={str(skip_vllm_install).lower()}")
    print(f"export SKIP_RAY_CLUSTER={str(skip_ray_cluster).lower()}")
    print(f"export SKIP_VLLM_START={str(skip_vllm_start).lower()}")
    print(f"export TEST_SERVICES={str(test_services).lower()}")
    print(f"export TEST_TIMEOUT={test_timeout}")
    
    # Export nodes as JSON for parsing
    print(f"export NODES_JSON='{json.dumps(nodes)}'")
    
except Exception as e:
    print(f"# Error loading config: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
)"

# Check if nodes are configured
if [ -z "$NODES_JSON" ] || [ "$NODES_JSON" = "[]" ]; then
    echo -e "${RED}Error: No nodes configured in config.yaml${NC}"
    echo "Please add at least one node with floating_ip"
    exit 1
fi

# Parse nodes
HEAD_NODE=""
WORKER_NODES=()

while IFS= read -r node; do
    ip=$(echo "$node" | python3 -c "import sys, json; print(json.load(sys.stdin).get('floating_ip', ''))" 2>/dev/null)
    role=$(echo "$node" | python3 -c "import sys, json; print(json.load(sys.stdin).get('role', ''))" 2>/dev/null)
    name=$(echo "$node" | python3 -c "import sys, json; print(json.load(sys.stdin).get('name', ''))" 2>/dev/null)
    user=$(echo "$node" | python3 -c "import sys, json; print(json.load(sys.stdin).get('username', 'ubuntu'))" 2>/dev/null)
    
    if [[ "$ip" == "<"*">" ]] || [ -z "$ip" ]; then
        echo -e "${RED}Error: Node $name has invalid floating_ip. Please fill in the Floating IP in config.yaml${NC}"
        exit 1
    fi
    
    if [ "$role" = "head" ]; then
        HEAD_NODE="$user@$ip"
        HEAD_NODE_NAME="$name"
    else
        WORKER_NODES+=("$user@$ip")
    fi
done < <(echo "$NODES_JSON" | python3 -c "import sys, json; [print(json.dumps(n)) for n in json.load(sys.stdin)]")

if [ -z "$HEAD_NODE" ]; then
    echo -e "${RED}Error: No head node found. Please set at least one node with role: head${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration loaded:${NC}"
echo "  Head node: $HEAD_NODE"
echo "  Worker nodes: ${#WORKER_NODES[@]}"
echo ""

# SSH helper function
ssh_cmd() {
    local host=$1
    local cmd=$2
    local ssh_opts="-i $SSH_KEY -o StrictHostKeyChecking=$([ "$SSH_STRICT" = "true" ] && echo "yes" || echo "no") -o UserKnownHostsFile=/dev/null"
    
    if [ "$SSH_STRICT" = "false" ]; then
        # Remove old host key if exists
        ssh-keygen -R "${host#*@}" 2>/dev/null || true
    fi
    
    ssh $ssh_opts "$host" "$cmd"
}

# Check SSH connectivity
echo -e "${YELLOW}Checking SSH connectivity...${NC}"
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}Error: SSH key not found: $SSH_KEY${NC}"
    exit 1
fi

for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    echo -n "  Testing $node... "
    if ssh_cmd "$node" "echo 'OK'" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        echo -e "${RED}Error: Cannot connect to $node${NC}"
        echo "Please check:"
        echo "  1. Floating IP is correct"
        echo "  2. SSH key path is correct: $SSH_KEY"
        echo "  3. Security group allows SSH (port 22)"
        exit 1
    fi
done
echo ""

# Step 1: Install NVIDIA Drivers (following howtodeplyray.md)
if [ "$SKIP_DRIVER" != "true" ] && [ "$DRIVER_INSTALL" = "true" ]; then
    echo -e "${YELLOW}Step 1: Installing NVIDIA drivers on all nodes...${NC}"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo "  Installing on $node..."
        # Execute commands as per howtodeplyray.md Step 3 (separate commands to avoid apt lock)
        ssh_cmd "$node" "sudo apt update"
        ssh_cmd "$node" "sudo apt upgrade -y"
        ssh_cmd "$node" "sudo add-apt-repository ppa:graphics-drivers/ppa -y"
        ssh_cmd "$node" "sudo apt install -y nvidia-driver-${DRIVER_VERSION}"
    done
    
    # Verify GPU (nvidia-smi as per howtodeplyray.md)
    echo "Verifying GPU installation..."
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo -n "  $node: "
        if ssh_cmd "$node" "nvidia-smi" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ GPU detected${NC}"
        else
            echo -e "${RED}✗ GPU not detected${NC}"
        fi
    done
    echo ""
fi

# Step 2: Verify CUDA via PyTorch (following howtodeplyray.md)
echo -e "${YELLOW}Step 2: Installing pip3, git and PyTorch with CUDA support...${NC}"
for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    echo "  Installing on $node..."
    ssh_cmd "$node" "sudo apt install -y python3-pip git"
    ssh_cmd "$node" "pip3 install --upgrade pip"
    ssh_cmd "$node" "pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX"
done

# Verify CUDA (as per howtodeplyray.md)
echo "Verifying CUDA..."
for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    echo -n "  $node: "
    result=$(ssh_cmd "$node" "python3 -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available()); print(\"Num GPUs:\", torch.cuda.device_count())'")
    echo "$result"
done
echo ""

# Step 3: Install Ray and vLLM (following howtodeplyray.md)
if [ "$SKIP_RAY_INSTALL" != "true" ]; then
    echo -e "${YELLOW}Step 3: Installing Ray and vLLM...${NC}"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo "  Installing on $node..."
        ssh_cmd "$node" "pip3 install 'ray[default]'"
        ssh_cmd "$node" "pip3 install vllm"
    done
    echo ""
fi

# Step 4: Start Ray Cluster (following howtodeplyray.md)
if [ "$SKIP_RAY_CLUSTER" != "true" ]; then
    echo -e "${YELLOW}Step 4: Starting Ray cluster...${NC}"
    
    # Get head node private IP (as per howtodeplyray.md)
    echo "  Getting head node private IP..."
    HEAD_PRIVATE_IP=$(ssh_cmd "$HEAD_NODE" "hostname -I | awk '{print \$1}'")
    echo "  Head node private IP: $HEAD_PRIVATE_IP"
    
    # Start Ray head (as per howtodeplyray.md)
    echo "  Starting Ray head on $HEAD_NODE..."
    ssh_cmd "$HEAD_NODE" "export PATH=\"\$HOME/.local/bin:\$PATH\" && ray start --head --port=$RAY_HEAD_PORT --dashboard-host=$RAY_DASHBOARD_HOST"
    
    # Start Ray workers (as per howtodeplyray.md)
    for worker in "${WORKER_NODES[@]}"; do
        echo "  Starting Ray worker on $worker..."
        ssh_cmd "$worker" "export PATH=\"\$HOME/.local/bin:\$PATH\" && ray start --address='$HEAD_PRIVATE_IP:$RAY_HEAD_PORT'"
    done
    
    # Verify cluster (as per howtodeplyray.md)
    echo "  Verifying Ray cluster..."
    result=$(ssh_cmd "$HEAD_NODE" "python3 -c 'import ray; ray.init(address=\"auto\"); print(\"Cluster resources:\", ray.cluster_resources())'")
    echo "  $result"
    echo ""
fi

# Step 5: Start vLLM on Each GPU Node (following howtodeplyray.md)
if [ "$SKIP_VLLM_START" != "true" ]; then
    echo -e "${YELLOW}Step 5: Starting vLLM services on each node...${NC}"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo "  Starting vLLM on $node..."
        # Start vLLM in background as per howtodeplyray.md
        ssh_cmd "$node" "CUDA_VISIBLE_DEVICES=$VLLM_GPU_PER_NODE python3 -m vllm.entrypoints.openai.api_server --model $VLLM_MODEL --port $VLLM_PORT --max-model-len $VLLM_MAX_LEN &"
        echo -e "    ${GREEN}✓ vLLM started${NC}"
    done
    echo ""
fi

# Step 6: Test Each vLLM Instance (following howtodeplyray.md)
if [ "$TEST_SERVICES" = "true" ]; then
    echo -e "${YELLOW}Step 6: Testing vLLM services...${NC}"
    node_num=1
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        ip="${node#*@}"
        echo "  Testing $ip:$VLLM_PORT (node $node_num)..."
        response=$(curl -s --max-time $TEST_TIMEOUT "http://$ip:$VLLM_PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$VLLM_MODEL\", \"prompt\": \"Hello from node $node_num\", \"max_tokens\": 16}" || echo "ERROR")
        
        if echo "$response" | grep -q "choices"; then
            echo -e "    ${GREEN}✓ Service responding${NC}"
        else
            echo -e "    ${YELLOW}⚠ Service may still be starting${NC}"
        fi
        node_num=$((node_num + 1))
    done
    echo ""
fi

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get head node floating IP for dashboard URL
HEAD_FLOATING_IP="${HEAD_NODE#*@}"
echo -e "${GREEN}Ray Dashboard:${NC}"
echo "  http://${HEAD_FLOATING_IP}:${RAY_DASHBOARD_PORT}/"
echo ""

echo -e "${GREEN}vLLM API Endpoints:${NC}"
for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    ip="${node#*@}"
    echo "  - http://$ip:$VLLM_PORT"
done
echo ""

