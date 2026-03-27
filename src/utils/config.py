"""Configuration Loader"""

import os
from typing import Any, Dict, Optional
import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        OmegaConf DictConfig
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def save_config(config: DictConfig, save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, "w") as f:
        OmegaConf.save(config, f)
        

def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs (later configs override earlier)."""
    return OmegaConf.merge(*configs)


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """Get nested config value using dot notation."""
    keys = key.split(".")
    value = config
    for k in keys:
        if hasattr(value, k):
            value = getattr(value, k)
        elif isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value
