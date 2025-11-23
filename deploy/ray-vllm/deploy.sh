#!/bin/bash

# Automated Multi-Node Ray Cluster + vLLM Deployment Script
# This script automates the entire deployment process described in howtodeplyray.md

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

# Step 1: Install NVIDIA Drivers
if [ "$SKIP_DRIVER" != "true" ] && [ "$DRIVER_INSTALL" = "true" ]; then
    echo -e "${YELLOW}Step 1: Installing NVIDIA drivers on all nodes...${NC}"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo "  Installing on $node..."
        ssh_cmd "$node" "sudo apt update && sudo apt upgrade -y && sudo add-apt-repository ppa:graphics-drivers/ppa -y && sudo apt update && sudo apt install -y nvidia-driver-${DRIVER_VERSION}"
    done
    
    if [ "$DRIVER_REBOOT" = "true" ]; then
        echo -e "${YELLOW}Rebooting nodes (this will take a few minutes)...${NC}"
        for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
            echo "  Rebooting $node..."
            ssh_cmd "$node" "sudo reboot" || true
        done
        
        echo "Waiting 60 seconds for nodes to reboot..."
        sleep 60
        
        echo "Waiting for nodes to come back online..."
        for i in {1..30}; do
            if ssh_cmd "$HEAD_NODE" "echo 'OK'" >/dev/null 2>&1; then
                echo -e "${GREEN}Nodes are back online${NC}"
                break
            fi
            echo -n "."
            sleep 10
        done
        echo ""
    fi
    
    # Verify GPU
    echo "Verifying GPU installation..."
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo -n "  $node: "
        if ssh_cmd "$node" "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null | head -1; then
            echo -e "${GREEN}✓ GPU detected${NC}"
        else
            echo -e "${RED}✗ GPU not detected${NC}"
        fi
    done
    echo ""
fi

# Step 2: Install Python pip (required before PyTorch)
echo -e "${YELLOW}Step 2: Installing Python pip...${NC}"
for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    echo "  Installing on $node..."
    ssh_cmd "$node" "sudo apt install -y python3-pip && pip3 install --upgrade pip"
done
echo ""

# Step 3: Install PyTorch
echo -e "${YELLOW}Step 3: Installing PyTorch with CUDA support...${NC}"
for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    echo "  Installing on $node..."
    ssh_cmd "$node" "pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX"
done

# Verify CUDA
echo "Verifying CUDA..."
for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    echo -n "  $node: "
    result=$(ssh_cmd "$node" "python3 -c 'import torch; print(\"CUDA:\", torch.cuda.is_available(), \"GPUs:\", torch.cuda.device_count())'")
    echo "$result"
done
echo ""

# Step 4: Install Ray and vLLM
if [ "$SKIP_RAY_INSTALL" != "true" ]; then
    echo -e "${YELLOW}Step 4: Installing Ray...${NC}"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo "  Installing on $node..."
        ssh_cmd "$node" "sudo apt install -y git && pip3 install 'ray[default]'"
    done
    
    # Verify Ray installation
    echo "Verifying Ray installation..."
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo -n "  $node: "
        if ssh_cmd "$node" "python3 -c 'import ray; print(\"Ray version:\", ray.__version__)'" 2>/dev/null; then
            echo -e "${GREEN}✓ Ray installed${NC}"
        else
            echo -e "${RED}✗ Ray installation failed${NC}"
            exit 1
        fi
    done
    echo ""
fi

if [ "$SKIP_VLLM_INSTALL" != "true" ]; then
    echo -e "${YELLOW}Step 5: Installing vLLM...${NC}"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo "  Installing on $node..."
        ssh_cmd "$node" "pip3 install vllm"
    done
fi

# Step 6: Start Ray Cluster
if [ "$SKIP_RAY_CLUSTER" != "true" ]; then
    echo -e "${YELLOW}Step 6: Starting Ray cluster...${NC}"
    
    # Get head node private IP
    echo "  Getting head node private IP..."
    HEAD_PRIVATE_IP=$(ssh_cmd "$HEAD_NODE" "hostname -I | awk '{print \$1}'")
    echo "  Head node private IP: $HEAD_PRIVATE_IP"
    
    # Start Ray head
    echo "  Starting Ray head on $HEAD_NODE..."
    # Ensure PATH includes ~/.local/bin where ray is typically installed
    ssh_cmd "$HEAD_NODE" "export PATH=\"\$HOME/.local/bin:\$PATH\" && ray stop" 2>/dev/null || true
    ssh_cmd "$HEAD_NODE" "export PATH=\"\$HOME/.local/bin:\$PATH\" && ray start --head --port=$RAY_HEAD_PORT --dashboard-host=$RAY_DASHBOARD_HOST --dashboard-port=$RAY_DASHBOARD_PORT"
    
    # Start Ray workers
    for worker in "${WORKER_NODES[@]}"; do
        echo "  Starting Ray worker on $worker..."
        ssh_cmd "$worker" "export PATH=\"\$HOME/.local/bin:\$PATH\" && ray stop" 2>/dev/null || true
        ssh_cmd "$worker" "export PATH=\"\$HOME/.local/bin:\$PATH\" && ray start --address='$HEAD_PRIVATE_IP:$RAY_HEAD_PORT'"
    done
    
    # Verify cluster
    echo "  Verifying Ray cluster..."
    result=$(ssh_cmd "$HEAD_NODE" "python3 -c 'import ray; ray.init(address=\"auto\"); print(ray.cluster_resources())'")
    echo "  Cluster resources: $result"
    echo ""
fi

# Step 7: Start vLLM services
if [ "$SKIP_VLLM_START" != "true" ]; then
    echo -e "${YELLOW}Step 7: Starting vLLM services...${NC}"
    echo -e "${YELLOW}Note: vLLM startup may take several minutes while loading the model...${NC}"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        echo "  Starting vLLM on $node..."
        # Stop existing vLLM if running
        ssh_cmd "$node" "pkill -f 'vllm.entrypoints.openai.api_server' || true"
        sleep 1
        
        # Start vLLM in background with proper nohup
        echo "    Command: CUDA_VISIBLE_DEVICES=$VLLM_GPU_PER_NODE python3 -m vllm.entrypoints.openai.api_server --model $VLLM_MODEL --port $VLLM_PORT --max-model-len $VLLM_MAX_LEN"
        # Create a startup script on remote node and execute it in background
        local ssh_opts="-i $SSH_KEY -o StrictHostKeyChecking=$([ "$SSH_STRICT" = "true" ] && echo "yes" || echo "no") -o UserKnownHostsFile=/dev/null"
        if [ "$SSH_STRICT" = "false" ]; then
            ssh-keygen -R "${node#*@}" 2>/dev/null || true
        fi
        
        # Create startup script on remote node
        ssh $ssh_opts "$node" "cat > /tmp/start_vllm.sh << 'SCRIPT_EOF'
#!/bin/bash
cd /tmp
exec nohup env CUDA_VISIBLE_DEVICES=$VLLM_GPU_PER_NODE python3 -m vllm.entrypoints.openai.api_server --model $VLLM_MODEL --port $VLLM_PORT --max-model-len $VLLM_MAX_LEN > /tmp/vllm.log 2>&1 &
SCRIPT_EOF
chmod +x /tmp/start_vllm.sh" || {
            echo -e "    ${RED}Failed to create startup script on $node${NC}"
            continue
        }
        
        # Execute script in background using SSH -f -n (background + no stdin)
        ssh $ssh_opts -f -n "$node" "/tmp/start_vllm.sh" || {
            echo -e "    ${RED}Failed to start vLLM on $node${NC}"
            echo "    Check logs: ssh -i $SSH_KEY $node 'tail -20 /tmp/vllm.log'"
            continue
        }
        echo -e "    ${GREEN}✓ vLLM process started (checking status...)${NC}"
        sleep 2
        
        # Verify process started
        if ssh_cmd "$node" "pgrep -f 'vllm.entrypoints.openai.api_server' > /dev/null"; then
            echo -e "    ${GREEN}✓ Process confirmed running${NC}"
        else
            echo -e "    ${YELLOW}⚠ Process may have failed to start, check logs${NC}"
        fi
    done
    
    echo ""
    echo "Waiting for vLLM services to initialize (this may take 1-5 minutes for model loading)..."
    echo "You can monitor progress by checking logs on each node:"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        ip="${node#*@}"
        echo "  ssh -i $SSH_KEY $node 'tail -f /tmp/vllm.log'"
    done
    echo ""
    
    # Wait and check services
    for i in {1..12}; do
        echo -n "  Checking services (attempt $i/12)... "
        all_running=true
        for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
            ip="${node#*@}"
            if ! curl -s --max-time 5 "http://$ip:$VLLM_PORT/health" >/dev/null 2>&1 && ! ssh_cmd "$node" "pgrep -f 'vllm.entrypoints.openai.api_server' > /dev/null"; then
                all_running=false
                break
            fi
        done
        
        if [ "$all_running" = "true" ]; then
            echo -e "${GREEN}✓ All services responding${NC}"
            break
        else
            echo "still starting..."
            sleep 10
        fi
    done
    
    # Final check
    echo ""
    echo "Final status check:"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        ip="${node#*@}"
        echo -n "  $ip:$VLLM_PORT... "
        if curl -s --max-time 5 "http://$ip:$VLLM_PORT/health" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Service responding${NC}"
        elif ssh_cmd "$node" "pgrep -f 'vllm.entrypoints.openai.api_server' > /dev/null"; then
            echo -e "${YELLOW}⚠ Process running but not yet ready (model still loading)${NC}"
        else
            echo -e "${RED}✗ Service not running${NC}"
            echo "    Check logs: ssh -i $SSH_KEY $node 'tail -50 /tmp/vllm.log'"
        fi
    done
    echo ""
fi

# Step 8: Test services
if [ "$TEST_SERVICES" = "true" ]; then
    echo -e "${YELLOW}Step 8: Testing vLLM services...${NC}"
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        ip="${node#*@}"
        echo "  Testing $ip:$VLLM_PORT..."
        response=$(curl -s --max-time $TEST_TIMEOUT "http://$ip:$VLLM_PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$VLLM_MODEL\", \"prompt\": \"Hello from $(hostname)\", \"max_tokens\": 16}" || echo "ERROR")
        
        if echo "$response" | grep -q "choices"; then
            echo -e "    ${GREEN}✓ Service responding${NC}"
        else
            echo -e "    ${YELLOW}⚠ Service may still be starting (check logs with: ssh -i $SSH_KEY $node 'tail -f /tmp/vllm.log')${NC}"
        fi
    done
    echo ""
fi

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Ray Dashboard: http://${HEAD_NODE#*@}:$RAY_DASHBOARD_PORT"
echo ""
echo "vLLM API Endpoints:"
for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    ip="${node#*@}"
    echo "  - http://$ip:$VLLM_PORT"
done
echo ""
echo "To check logs:"
for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
    echo "  ssh -i $SSH_KEY $node 'tail -f /tmp/vllm.log'"
done
echo ""

