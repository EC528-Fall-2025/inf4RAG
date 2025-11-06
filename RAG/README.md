# RAG Service (Two-Container Setup)

Minimal instructions for running the Retrieval-Augmented Generation (RAG) API with a Qdrant vector database. This setup uses only local Sentence-Transformers embeddings (no external API keys).

## Endpoints
- `POST /rag/upload` — Upload a zip of documents and build an index
- `POST /rag/query` — Query a dataset and get retrieved chunks + prompt
- `GET  /rag/health` — Health check

## Build Image
```bash
podman build -t inf4rag/rag:latest -f RAG/Dockerfile .
# or
docker build -t inf4rag/rag:latest -f RAG/Dockerfile .
```

## Run With Two Containers (Qdrant + RAG)
You only need two containers if `RAG_BACKEND=qdrant`. Qdrant stores embeddings; RAG API performs ingest/query.

### Podman (macOS)
Create a network:
```bash
podman network create rag-net
```
Start Qdrant:
```bash
podman run -d --name qdrant --network rag-net -p 6333:6333 qdrant/qdrant:latest
```
Start RAG API:
```bash
podman run --rm --name rag --network rag-net -p 8001:8001 \
  -e RAG_BACKEND=qdrant \
  -e RAG_EMBEDDING_PROVIDER=local \
  -e RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  -e QDRANT_URL=http://qdrant:6333 \
  -v "$(pwd)/persistent_data:/data:Z" \
  inf4rag/rag:latest
```

### Docker Alternative
```bash
docker network create rag-net

docker run -d --name qdrant --network rag-net -p 6333:6333 qdrant/qdrant:latest

docker run --rm --name rag --network rag-net -p 8001:8001 \
  -e RAG_BACKEND=qdrant \
  -e RAG_EMBEDDING_PROVIDER=local \
  -e RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  -e QDRANT_URL=http://qdrant:6333 \
  -v "$(pwd)/persistent_data:/data" \
  inf4rag/rag:latest
```

### Without Custom Network (Host Mapping)
Podman:
```bash
podman run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
podman run --rm -p 8001:8001 \
  -e RAG_BACKEND=qdrant \
  -e RAG_EMBEDDING_PROVIDER=local \
  -e RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  -e QDRANT_URL=http://host.containers.internal:6333 \
  -v "$(pwd)/persistent_data:/data:Z" \
  inf4rag/rag:latest
```
Docker:
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
docker run --rm -p 8001:8001 \
  -e RAG_BACKEND=qdrant \
  -e RAG_EMBEDDING_PROVIDER=local \
  -e RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -v "$(pwd)/persistent_data:/data" \
  inf4rag/rag:latest
```

## Test Workflow
Create a sample dataset (two small text files), upload, then query:

Please run
```python
python3 /Users/matthewkweon/Documents/GitHub/inf4RAG/tmp_repos_ec528/fetch_and_fill_readmes.py
```
then run this to zip the file:
```bash
zip -r repos-ec528-10-enriched.zip tmp_repos_ec528 -x 'tmp_repos_ec528/README-template.txt' 'tmp_repos_ec528/*.py' '*/.DS_Store'
```

Run the following commands to test the functionality with rag using the 10 repos of the projects up to date.
```bash

curl -s http://localhost:8001/rag/health | jq

curl -s -F "file=@repos-ec528-10-enriched.zip" -F "dataset_id=ec528-repos" \
  http://localhost:8001/rag/upload | jq

curl -s -X POST http://localhost:8001/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the inf4rag project about?","dataset_id":"ec528-repos","top_k":3}' | jq
```

## Environment Variables (Used Here)
- `RAG_BACKEND`: set to `qdrant` for two-container mode; use `faiss` for single-container local index.
- `RAG_PERSIST_DIR`: path for datasets (container default `/data`).
- `RAG_EMBEDDING_PROVIDER`: keep as `local`.
- `RAG_EMBEDDING_MODEL`: sentence-transformers model.
- `QDRANT_URL`: URL of the Qdrant service.

## Cleanup
```bash
podman rm -f rag qdrant || true
podman network rm rag-net || true
```

## Single-Container Option
Use `RAG_BACKEND=faiss` and omit all Qdrant variables; no Qdrant container needed.