# 1 Configure backend


Edit chatbotbasic/WebChat/config.yaml. The base_url below contains the maintainer’s IP/port. Teammates must change it to their own vLLM endpoint.


vllm:
  base_url: "http://199.94.60.224:8000/v1"   # <— CHANGE for your deployment (or use 127.0.0.1 if tunneling)
  api_key: "ec528"                            # <— CHANGE if your vLLM uses a different key
  model: "qwen-4b-instruct"                   # <— served-model-name (or HF id) that your vLLM exposes

rag:
  enabled_by_default: true
  docs_dir: "./data/docs"                     # drop .txt/.md/.pdf here if you want RAG
  index_dir: "./data/index/faiss"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

server:
  host: "0.0.0.0"
  port: 7861
  log_level: "info"




If you don’t control the public security group or prefer not to open ports, you may use SSH tunneling from your laptop and point base_url to http://127.0.0.1:8000/v1:


ssh -i "<path-to-key>.pem" -L 8000:127.0.0.1:8000 ubuntu@<your-openstack-ip>


# 2 Ensure vLLM is running on your GPU server

source /data/venvs/vllm/bin/activate
VLLM_USE_V1=1 \
vllm serve /data/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen-4b-instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key ec528 \
  --max-model-len 128000 --gpu-memory-utilization 0.95



# 3 Connectivity check from your laptop (PowerShell)

$headers = @{ Authorization = "Bearer ec528" }
Invoke-RestMethod -Uri "http://<your-ip>:8000/v1/models" -Method GET -Headers $headers
# or with real curl on Windows:
curl.exe -i "http://<your-ip>:8000/v1/models" -H "Authorization: Bearer ec528"


# 4 Run the backend

cd chatbotbasic/WebChat
python app.py


# 5 Chat with the model (no RAG / pure LLM)
$body = @{
    user_input = "Say hi and tell me today's date."
    provider   = "vllm"
    use_rag    = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:7861/chat" -Method Post -ContentType "application/json" -Body $body

