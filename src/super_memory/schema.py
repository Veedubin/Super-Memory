"""LanceDB schema and embedding model for Super-Memory."""

import datetime
from typing import Optional

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from .config import get_config


config = get_config()

registry = get_registry().get("sentence-transformers")
embed_model = registry.create(name=config.model, device=config.device)


class MemorySchema(LanceModel):
    """Schema for memory entries in LanceDB."""

    text: str = embed_model.SourceField()
    vector: Vector(embed_model.ndims()) = embed_model.VectorField()  # type: ignore
    source_type: str = "session"
    source_path: Optional[str] = None
    timestamp: datetime.datetime
    content_hash: Optional[str] = None
    metadata_json: Optional[str] = None
