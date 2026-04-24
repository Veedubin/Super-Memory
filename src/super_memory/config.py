"""Configuration system for Super-Memory.

Environment variables:
- SUPER_MEMORY_DB_PATH: Path to LanceDB database (default: ./memory_data)
- SUPER_MEMORY_DEVICE: Device for embedding model (default: auto, options: auto, cpu, cuda)
- SUPER_MEMORY_MODEL: MiniLM sentence transformer model name (default: sentence-transformers/all-MiniLM-L6-v2)
- EMBEDDING_STRATEGY: "PARALLEL", "TIERED", or "MINILM_ONLY" (default: "TIERED")
- BGE_THRESHOLD: float for tiered fallback threshold (default: 0.72)
- BGE_MODEL: BGE model name for long-form embeddings (default: BAAI/bge-large-en-v1.5)
- AUTO_SUMMARIZE_INTERVAL: integer for auto-summarize interval (default: 15)
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
    embedding_strategy: str
    bge_threshold: float
    bge_model: str
    auto_summarize_interval: int


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get cached configuration from environment variables.

    Returns:
        Config instance with validated settings.

    Raises:
        ValueError: If SUPER_MEMORY_DEVICE is not one of {auto, cpu, cuda}.
        ValueError: If EMBEDDING_STRATEGY is not one of {PARALLEL, TIERED, MINILM_ONLY}.
    """
    raw_path = os.environ.get("SUPER_MEMORY_DB_PATH", "./memory_data")
    db_path = os.path.abspath(raw_path)

    model = os.environ.get(
        "SUPER_MEMORY_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    device_raw = os.environ.get("SUPER_MEMORY_DEVICE", "auto")
    dtype = os.environ.get("SUPER_MEMORY_DTYPE", "float32")

    # New dual-embedding config options
    embedding_strategy = os.environ.get("EMBEDDING_STRATEGY", "TIERED").upper()
    valid_strategies = {"PARALLEL", "TIERED", "MINILM_ONLY"}
    if embedding_strategy not in valid_strategies:
        raise ValueError(
            f"Invalid EMBEDDING_STRATEGY '{embedding_strategy}'. "
            f"Must be one of: {valid_strategies}"
        )

    bge_threshold = float(os.environ.get("BGE_THRESHOLD", "0.72"))
    bge_model = os.environ.get("BGE_MODEL", "BAAI/bge-large-en-v1.5")
    auto_summarize_interval = int(os.environ.get("AUTO_SUMMARIZE_INTERVAL", "15"))

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

    return Config(
        db_path=db_path,
        device=device,
        model=model,
        dtype=dtype,
        embedding_strategy=embedding_strategy,
        bge_threshold=bge_threshold,
        bge_model=bge_model,
        auto_summarize_interval=auto_summarize_interval,
    )


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
