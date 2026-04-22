"""LanceDB schema and embedding model for Super-Memory."""

import datetime
from typing import Optional

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from .config import get_config


config = get_config()

# Lazy initialization for embed models to avoid import-time failures
_embed_model_minilm = None
_embed_model_bge = None


def get_embed_model():
    """Lazily initialize and return the MiniLM embedding model.

    Returns:
        The MiniLM embedding model instance.
    """
    global _embed_model_minilm
    if _embed_model_minilm is None:
        registry = get_registry().get("sentence-transformers")
        _embed_model_minilm = registry.create(name=config.model, device=config.device)

        # Convert to FP16 if configured
        if config.dtype == "float16":
            _embed_model_minilm = _embed_model_minilm.half()
    return _embed_model_minilm


def get_bge_embed_model():
    """Lazily initialize and return the BGE embedding model.

    Returns:
        The BGE embedding model instance.
    """
    global _embed_model_bge
    if _embed_model_bge is None:
        registry = get_registry().get("sentence-transformers")
        _embed_model_bge = registry.create(name=config.bge_model, device=config.device)

        # Convert to FP16 if configured
        if config.dtype == "float16":
            _embed_model_bge = _embed_model_bge.half()
    return _embed_model_bge


# We cannot create schemas at module level because it would eagerly
# initialize the embedding models. Instead, we create them lazily when first accessed.

_cached_schema = None
_cached_schema_long = None


def _get_memory_schema():
    """Get or create the MemorySchema class (MiniLM, 384-dim).

    This function enables lazy initialization of the embedding model, so that
    imports of this module don't fail if model files are missing.

    Returns:
        MemorySchema class with proper embedding fields.
    """
    global _cached_schema
    if _cached_schema is None:
        _cached_schema = _create_memory_schema()
    return _cached_schema


def _get_memory_schema_long():
    """Get or create the MemorySchemaLong class (BGE-Large, 1024-dim).

    This function enables lazy initialization of the BGE embedding model, so that
    the heavy model is only loaded when first needed.

    Returns:
        MemorySchemaLong class with proper embedding fields.
    """
    global _cached_schema_long
    if _cached_schema_long is None:
        _cached_schema_long = _create_memory_schema_long()
    return _cached_schema_long


def _create_memory_schema():
    """Create the MemorySchema class with properly initialized embed model.

    This function should be called only when the schema is actually needed,
    not at import time.

    Returns:
        MemorySchema class with proper embedding fields.
    """
    embed = get_embed_model()

    class MemorySchema(LanceModel):
        """Schema for memory entries in LanceDB using MiniLM (384-dim)."""

        text: str = embed.SourceField()
        vector: Vector(embed.ndims()) = embed.VectorField()  # type: ignore
        source_type: str = "session"
        source_path: Optional[str] = None
        timestamp: datetime.datetime
        content_hash: Optional[str] = None
        metadata_json: Optional[str] = None

    return MemorySchema


def _create_memory_schema_long():
    """Create the MemorySchemaLong class with properly initialized BGE embed model.

    This function should be called only when the schema is actually needed,
    not at import time.

    Returns:
        MemorySchemaLong class with proper embedding fields.
    """
    embed = get_bge_embed_model()

    class MemorySchemaLong(LanceModel):
        """Schema for memory entries in LanceDB using BGE-Large (1024-dim)."""

        text: str = embed.SourceField()
        vector: Vector(embed.ndims()) = embed.VectorField()  # type: ignore
        source_type: str = "session"
        source_path: Optional[str] = None
        timestamp: datetime.datetime
        content_hash: Optional[str] = None
        metadata_json: Optional[str] = None

    return MemorySchemaLong


def __getattr__(name):
    """Module-level attribute access for lazy schema initialization."""
    if name == "MemorySchema":
        return _get_memory_schema()
    if name == "MemorySchemaLong":
        return _get_memory_schema_long()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes including lazy-loaded MemorySchema."""
    return list(globals().keys()) + ["MemorySchema", "MemorySchemaLong"]
