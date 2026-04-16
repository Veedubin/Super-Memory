"""Configuration system for Super-Memory.

Environment variables:
- SUPER_MEMORY_DB_PATH: Path to LanceDB database (default: ./memory_data)
- SUPER_MEMORY_DEVICE: Device for embedding model (default: auto, options: auto, cpu, cuda)
- SUPER_MEMORY_MODEL: Sentence transformer model name (default: BAAI/bge-large-en-v1.5)
"""

from dataclasses import dataclass
from functools import lru_cache
import os

import torch


@dataclass(frozen=True)
class Config:
    """Immutable configuration for Super-Memory."""

    db_path: str
    device: str
    model: str


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get cached configuration from environment variables.

    Returns:
        Config instance with validated settings.

    Raises:
        ValueError: If SUPER_MEMORY_DEVICE is not one of {auto, cpu, cuda}.
    """
    db_path = os.environ.get("SUPER_MEMORY_DB_PATH", "./memory_data")
    model = os.environ.get("SUPER_MEMORY_MODEL", "BAAI/bge-large-en-v1.5")
    device_raw = os.environ.get("SUPER_MEMORY_DEVICE", "auto")

    valid_devices = {"auto", "cpu", "cuda"}
    if device_raw not in valid_devices:
        raise ValueError(
            f"Invalid SUPER_MEMORY_DEVICE '{device_raw}'. "
            f"Must be one of: {valid_devices}"
        )

    if device_raw == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_raw

    return Config(db_path=db_path, device=device, model=model)
