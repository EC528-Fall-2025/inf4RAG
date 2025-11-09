#!/usr/bin/env python3
"""
simple_lb.py - Simple Round-Robin Load Balancer for vLLM Data Parallelism

This is a lightweight load balancer that distributes requests across multiple
vLLM instances in a round-robin fashion. Useful for testing data parallelism
without needing nginx or other complex load balancers.

Usage:
    python3 simple_lb.py --backends http://localhost:8001 http://localhost:8002 --port 8000
"""

import argparse
import asyncio
from itertools import cycle
from typing import List

import aiohttp
from quart import Quart, request, Response

app = Quart(__name__)

# Global state
backends: List[str] = []
backend_cycle = None
http_session = None

@app.before_serving
async def startup():
    """Initialize shared HTTP session with connection pooling."""
    global http_session
    connector = aiohttp.TCPConnector(limit=200)
    timeout = aiohttp.ClientTimeout(total=600)
    http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    print("✓ Load balancer started with connection pooling")

@app.after_serving
async def shutdown():
    """Clean up HTTP session."""
    global http_session
    if http_session:
        await http_session.close()
    print("✓ Load balancer shut down")

async def forward_to_backend(backend_url: str, path: str, method: str):
    """Forward request to a backend server."""
    if not http_session:
        return {"error": "HTTP session not initialized"}, 500
    
    url = f"{backend_url}{path}"
    
    try:
        # Prepare request data
        json_data = None
        if method in ["POST", "PUT", "PATCH"]:
            json_data = await request.get_json(silent=True)
        
        # Forward request
        async with http_session.request(
            method=method,
            url=url,
            json=json_data,
            headers={k: v for k, v in request.headers.items() if k.lower() != 'host'}
        ) as response:
            content = await response.read()
            
            # Create response with same status code and headers
            resp = Response(content, status=response.status)
            for key, value in response.headers.items():
                if key.lower() not in ['content-encoding', 'content-length', 'transfer-encoding']:
                    resp.headers[key] = value
            
            return resp
    
    except Exception as e:
        print(f"✗ Error forwarding to {backend_url}: {e}")
        return {"error": f"Backend error: {str(e)}"}, 502

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
async def load_balance(path):
    """Load balance requests across backends."""
    backend = next(backend_cycle)
    return await forward_to_backend(backend, f"/{path}", request.method)

@app.route('/', methods=['GET', 'POST'])
async def load_balance_root():
    """Handle root path requests."""
    backend = next(backend_cycle)
    return await forward_to_backend(backend, "/", request.method)

def main():
    parser = argparse.ArgumentParser(description='Simple Round-Robin Load Balancer for vLLM')
    parser.add_argument('--backends', nargs='+', required=True,
                        help='Backend URLs (e.g., http://localhost:8001 http://localhost:8002)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to listen on (default: 8000)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    global backends, backend_cycle
    backends = args.backends
    backend_cycle = cycle(backends)
    
    print(f"\n{'='*60}")
    print("Simple Load Balancer for vLLM Data Parallelism")
    print(f"{'='*60}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print(f"Backends:")
    for i, backend in enumerate(backends, 1):
        print(f"  {i}. {backend}")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
