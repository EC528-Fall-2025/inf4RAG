#!/usr/bin/env python3
"""
Test script for Ray Serve + vLLM service
"""

import requests
import json
import sys
import argparse


def test_service(base_url: str, prompt: str = "Hello, how are you?"):
    """
    Test Ray Serve + vLLM service
    
    Args:
        base_url: Service base URL (e.g., http://199.94.61.26:8000)
        prompt: Test prompt
    """
    # Build complete API endpoint
    api_url = f"{base_url}/VLLMDeployment"
    
    print(f"Testing service: {api_url}")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    # Build request payload
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0,
        "max_tokens": 100
    }
    
    try:
        # Send request
        print("Sending request...")
        response = requests.post(
            api_url,
            json=payload,
            timeout=60  # 60 second timeout
        )
        
        # Check response status
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        print("✓ Request successful!")
        print("\nResponse:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Extract generated text
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0].get("message", {}).get("content", "")
            if content:
                print("\nGenerated text:")
                print(content)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to service")
        print(f"   Please ensure the service is running: {base_url}")
        return False
        
    except requests.exceptions.Timeout:
        print("✗ Error: Request timeout")
        print("   Service may be processing, please retry later")
        return False
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP error: {e}")
        print(f"   Response content: {response.text}")
        return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_health(base_url: str):
    """
    Test service health status (if supported)
    
    Args:
        base_url: Service base URL
    """
    try:
        # Ray Serve doesn't provide health check endpoint by default, but can try root path
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Service status: {response.status_code}")
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ray Serve + vLLM service")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Service URL (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Test prompt (default: Hello, how are you?)"
    )
    
    args = parser.parse_args()
    
    # Test service
    success = test_service(args.url, args.prompt)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Test passed!")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ Test failed")
        sys.exit(1)

