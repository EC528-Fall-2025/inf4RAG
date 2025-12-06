"""
Load configuration from config.yaml
"""

import yaml
import os
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config.yaml file
    
    Returns:
        Dictionary containing configuration
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_path)
    
    if not os.path.exists(config_file):
        print(f"Warning: Config file not found: {config_file}")
        print("Using default values")
        return get_default_config()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Extract values with defaults
        model_path = config.get('model', {}).get('path', 'gpt2')
        num_replicas = config.get('deployment', {}).get('num_replicas')
        gpus_per_replica = config.get('deployment', {}).get('gpus_per_replica')
        tensor_parallel_size = config.get('deployment', {}).get('tensor_parallel_size')
        host = config.get('service', {}).get('host', '0.0.0.0')
        port = config.get('service', {}).get('port', 8000)
        
        return {
            'model_path': model_path,
            'num_replicas': num_replicas,
            'gpus_per_replica': gpus_per_replica,
            'tensor_parallel_size': tensor_parallel_size,
            'host': host,
            'port': port
        }
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default values")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'model_path': 'gpt2',
        'num_replicas': None,
        'gpus_per_replica': None,
        'tensor_parallel_size': None,
        'host': '0.0.0.0',
        'port': 8000
    }


if __name__ == "__main__":
    # Test loading config
    config = load_config()
    print("Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")

