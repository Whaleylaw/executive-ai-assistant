"""Configuration utilities for the Executive AI Assistant."""

import os
import yaml
from typing import Dict, Any


def get_config(runtime_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get configuration, combining defaults with runtime config.
    
    Args:
        runtime_config: Runtime configuration to override defaults
        
    Returns:
        The combined configuration
    """
    # Load default configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Add memory flag (defaults to True)
    config["memory"] = True
    
    # Override with runtime config if provided
    if runtime_config and "configurable" in runtime_config:
        if "memory" in runtime_config["configurable"]:
            config["memory"] = runtime_config["configurable"]["memory"]
    
    return config
