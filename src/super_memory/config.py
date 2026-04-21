"""Configuration system for Super-Memory.

Environment variables:
- SUPER_MEMORY_DB_PATH: Path to LanceDB database (default: ./memory_data)
- SUPER_MEMORY_DEVICE: Device for embedding model (default: auto, options: auto, cpu, cuda)
- SUPER_MEMORY_MODEL: Sentence transformer model name (default: BAAI/bge-large-en-v1.5)
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
import os

import torch


@dataclass(frozen=True)
class Config:
    """Immutable configuration for Super-Memory."""

    db_path: str
    device: str
    model: str
    dtype: str


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get cached configuration from environment variables.

    Returns:
        Config instance with validated settings.

    Raises:
        ValueError: If SUPER_MEMORY_DEVICE is not one of {auto, cpu, cuda}.
    """
    raw_path = os.environ.get("SUPER_MEMORY_DB_PATH", "./memory_data")
    db_path = os.path.abspath(raw_path)
    model = os.environ.get(
        "SUPER_MEMORY_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    device_raw = os.environ.get("SUPER_MEMORY_DEVICE", "auto")
    dtype = os.environ.get("SUPER_MEMORY_DTYPE", "float32")

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

    return Config(db_path=db_path, device=device, model=model, dtype=dtype)


def configure_logging() -> None:
    """Configure root logging for Super-Memory.

    Reads SUPER_MEMORY_LOG_LEVEL env var (default: WARNING).
    Only configures if handlers are not already set (respects external config).
    """
    level_name = os.environ.get("SUPER_MEMORY_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)

    logger = logging.getLogger("super_memory")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)

    # Reduce noise from third-party libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
