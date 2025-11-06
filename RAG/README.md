# RAG Service (Flask)

This is a lightweight Retrieval-Augmented Generation (RAG) microservice that provides:

- `POST /rag/upload` — Upload a zip of documents to index
- `POST /rag/query` — Query your dataset and return retrieved chunks and a constructed prompt
- `GET  /rag/health` — Health check
- `GET  /rag/datasets` — List available datasets

It uses a pure NumPy TF–IDF index and serves a minimal Flask API.

## Configuration

Environment variables:

- `PORT` (default: `8001`): HTTP port.
- `RAG_PERSIST_DIR` (default: `/data` in container, `persistent_data` locally): where datasets and indexes are stored. Mount a volume here in containers.
- `RAG_DEBUG` (default: `0`): set to `1` to enable Flask debug.

The maximum upload size is enforced by the app and validator.

## Local run (without Docker)

```bash
# From repo root
cd RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PORT=8001
export RAG_PERSIST_DIR=$(pwd)/../persistent_data
python entrypoint.py
```

Test health:

```bash
curl -s http://localhost:8001/rag/health | jq
```

## Build container

```bash
# From repo root
DOCKER_BUILDKIT=1 docker build -t inf4rag/rag:latest -f RAG/Dockerfile .
```

Run it locally:

```bash
docker run --rm -p 8001:8001 \
	-e PORT=8001 \
	-e RAG_PERSIST_DIR=/data \
	-v $(pwd)/persistent_data:/data \
	inf4rag/rag:latest
```

Upload a zip for indexing:

```bash
curl -F "file=@/path/to/docs.zip" -F "dataset_id=my-dataset" \
	http://localhost:8001/rag/upload | jq
```

Query:

```bash
curl -s -X POST http://localhost:8001/rag/query \
	-H 'Content-Type: application/json' \
	-d '{"query":"What is inside?","dataset_id":"my-dataset","top_k":3}' | jq
```

## Kubernetes (Kustomize)

Resources are under `deploy/kubernetes/base`:

- `rag-deployment.yaml`: RAG deployment (uses image `ghcr.io/ec528-fall-2025/inf4rag-rag:latest` by default — change as needed)
- `rag-service.yaml`: ClusterIP service on port 8001
- `kustomization.yaml`: already includes RAG resources

Example apply (namespace and other infra must exist):

```bash
# Optional: set your image
# kustomize edit set image ghcr.io/ec528-fall-2025/inf4rag-rag:latest=<your-registry>/<your-image>:<tag>

kubectl apply -k deploy/kubernetes/base

kubectl -n vllm-inference get deploy,svc | grep rag
```

To persist data across pod restarts, replace the `emptyDir` volume in `rag-deployment.yaml` with a PVC.

## Podman usage

Podman is a drop-in replacement for Docker for most commands. Examples:

Build image:

```bash
podman build -t inf4rag/rag:latest -f RAG/Dockerfile .
```

Run container (rootless, macOS):

```bash
podman run --rm -p 8001:8001 \
	-e PORT=8001 \
	-e RAG_PERSIST_DIR=/data \
	-v $(pwd)/persistent_data:/data:Z \
	inf4rag/rag:latest
```

Notes:
- The trailing `:Z` label on the volume is helpful on SELinux-enabled systems (Linux). On macOS, it is harmless.
- If you use Podman machine on macOS, ensure your `podman machine` has the shared directory mounted (default is your home directory). If `$(pwd)` is outside the shared path, either move the project or reconfigure `podman machine`.

Push image to a registry (example GitHub Container Registry):

```bash
podman login ghcr.io
podman tag inf4rag/rag:latest ghcr.io/<org-or-user>/inf4rag-rag:latest
podman push ghcr.io/<org-or-user>/inf4rag-rag:latest
```

Optional: generate Kubernetes YAML from a running container config using Podman:

```bash
# Run in detached mode first
podman run -d --name rag \
	-p 8001:8001 \
	-e PORT=8001 -e RAG_PERSIST_DIR=/data \
	-v $(pwd)/persistent_data:/data:Z \
	inf4rag/rag:latest

# Generate k8s manifests
podman generate kube rag > podman-rag.yaml
```

## Notes

- The service is CPU-bound for indexing; scale replicas if throughput is needed (stateless API, but shared storage required for consistency).
- Keep uploads moderate in size; for very large corpora, move to chunked/upload streaming and batch ingestion.
# RAG Module

## Usage 

`python entrypoint.py`

## Configuration

`vim my_rag/config.py`