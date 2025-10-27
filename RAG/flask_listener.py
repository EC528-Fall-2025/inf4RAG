"""Flask application with minimal route handlers."""

from __future__ import annotations
from flask import Flask, request, jsonify
from pathlib import Path
import tempfile

from .config import RAGConfig
from .pipeline import RAGPipeline
from .services.upload_service import UploadService
from .services.query_service import QueryService
from .utils.validators import validate_upload_request, MAX_UPLOAD_SIZE


# Initialize services
cfg = RAGConfig()
pipeline = RAGPipeline(cfg)
upload_service = UploadService(cfg, pipeline)
query_service = QueryService(cfg, pipeline)

# Create Flask app
app = Flask(__name__)


@app.route("/rag/upload", methods=["POST"])
def upload_documents():
    """Upload endpoint - delegates to upload service."""
    # Validate request
    dataset_id, error = validate_upload_request(request)
    if error:
        return jsonify({"status": "error", "message": error}), 400
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            request.files['file'].save(tmp_file.name)
            tmp_path = Path(tmp_file.name)
        
        # Process upload
        result = upload_service.upload_and_ingest(tmp_path, dataset_id)
        
        # Clean up
        tmp_path.unlink()
        
        status_code = 200 if result["status"] == "success" else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Upload failed: {str(e)}"
        }), 500


@app.route("/rag/query", methods=["POST"])
def rag_query():
    """Query endpoint - delegates to query service."""
    data = request.get_json(force=True)
    
    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON data provided"
        }), 400
    
    # Extract parameters
    query = data.get("query")
    dataset_id = data.get("dataset_id", "default-dataset")
    top_k = int(data.get("top_k", cfg.default_top_k))
    
    # Execute query
    result = query_service.execute_query(query, dataset_id, top_k)
    
    status_code = 200 if result["status"] == "success" else 404
    return jsonify(result), status_code


@app.route("/rag/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "RAG Listener",
        "version": "1.0.0"
    }), 200


@app.route("/rag/datasets", methods=["GET"])
def list_datasets():
    """List all datasets - could also be moved to a service."""
    try:
        datasets_dir = cfg.persistent_dir / "datasets"
        if not datasets_dir.exists():
            return jsonify({
                "status": "success",
                "datasets": []
            }), 200
        
        datasets = []
        for dataset_id_dir in datasets_dir.iterdir():
            if dataset_id_dir.is_dir():
                meta_file = dataset_id_dir / "dataset.json"
                meta = {}
                
                if meta_file.exists():
                    try:
                        import orjson as json
                        meta = json.loads(meta_file.read_bytes())
                    except Exception:
                        meta = {"dataset_id": dataset_id_dir.name}
                
                # Check if index exists
                index_dir = dataset_id_dir / "index"
                has_index = index_dir.exists() and any(index_dir.iterdir())
                
                datasets.append({
                    "dataset_id": meta.get("dataset_id", dataset_id_dir.name),
                    "user_id": meta.get("user_id", "unknown"),
                    "dataset_name": meta.get("dataset_name", dataset_id_dir.name),
                    "has_index": has_index
                })
        
        return jsonify({
            "status": "success",
            "datasets": datasets
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to list datasets: {str(e)}"
        }), 500


if __name__ == "__main__":
    cfg.persistent_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Starting RAG Listener Service")
    print("=" * 60)
    print(f"Persistent data directory: {cfg.persistent_dir.resolve()}")
    print("Available endpoints:")
    print("  POST /rag/upload      - Upload and process zip documents")
    print("  POST /rag/query       - Query with RAG")
    print("  GET  /rag/health      - Health check")
    print("  GET  /rag/datasets    - List all datasets")
    print("=" * 60)
    print()
    
    app.run(host="0.0.0.0", port=8001, debug=True)