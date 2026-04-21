"""LanceDB schema and embedding model for Super-Memory."""

import datetime
from typing import Optional

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from .config import get_config


config = get_config()

# Lazy initialization for embed model to avoid import-time failures
_embed_model = None


def get_embed_model():
    """Lazily initialize and return the embedding model.

    Returns:
        The embedding model instance.
    """
    global _embed_model
    if _embed_model is None:
        from .config import get_config

        config = get_config()
        registry = get_registry().get("sentence-transformers")
        _embed_model = registry.create(name=config.model, device=config.device)

        # Convert to FP16 if configured
        if config.dtype == "float16":
            _embed_model = _embed_model.half()
    return _embed_model


# We cannot create MemorySchema at module level because it would eagerly
# initialize the embedding model. Instead, we create it lazily when first accessed.
# This requires careful handling since Python class definitions are evaluated at import.

_cached_schema = None


def _get_memory_schema():
    """Get or create the MemorySchema class.

    This function enables lazy initialization of the embedding model, so that
    imports of this module don't fail if model files are missing.

    Returns:
        MemorySchema class with proper embedding fields.
    """
    global _cached_schema
    if _cached_schema is None:
        _cached_schema = _create_memory_schema()
    return _cached_schema


def _create_memory_schema():
    """Create the MemorySchema class with properly initialized embed model.

    This function should be called only when the schema is actually needed,
    not at import time.

    Returns:
        MemorySchema class with proper embedding fields.
    """
    embed = get_embed_model()

    class MemorySchema(LanceModel):
        """Schema for memory entries in LanceDB."""

        text: str = embed.SourceField()
        vector: Vector(embed.ndims()) = embed.VectorField()  # type: ignore
        source_type: str = "session"
        source_path: Optional[str] = None
        timestamp: datetime.datetime
        content_hash: Optional[str] = None
        metadata_json: Optional[str] = None

    return MemorySchema


def __getattr__(name):
    """Module-level attribute access for lazy schema initialization."""
    if name == "MemorySchema":
        return _get_memory_schema()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes including lazy-loaded MemorySchema."""
    return list(globals().keys()) + ["MemorySchema"]
