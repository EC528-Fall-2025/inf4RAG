from __future__ import annotations
from flask import Flask, request, jsonify
from pathlib import Path

from .config import RAGConfig
from .file_io import allocate_dataset
from .pipeline import RAGPipeline

app = Flask(__name__)
cfg = RAGConfig()
pipeline = RAGPipeline(cfg)

@app.get("/rag/allocate-dir")
def allocate_dir():
    user_id = request.args.get("user_id", "u")
    dataset_name = request.args.get("dataset_name", "ds")
    dataset_id, ingest_dir = allocate_dataset(cfg, user_id, dataset_name)
    return jsonify({"dataset_id": dataset_id, "ingest_dir": str(ingest_dir.resolve())})

@app.post("/rag/ingest")
def ingest():
    data = request.get_json(force=True)
    dataset_id = data["dataset_id"]
    pipeline.ingest(dataset_id)
    return jsonify({"status": "ok", "dataset_id": dataset_id})

@app.post("/rag/generate")
def generate():
    data = request.get_json(force=True)
    dataset_id = data["dataset_id"]
    query = data["query"]
    k = int(data.get("top_k", cfg.default_top_k))
    result = pipeline.generate(dataset_id=dataset_id, query=query, top_k=k)
    return jsonify({
        "dataset_id": result.dataset_id,
        "query": result.query,
        "top_k": result.top_k,
        "retrieved": result.retrieved,
        "prompt": result.prompt
    })

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8001, debug=True)
