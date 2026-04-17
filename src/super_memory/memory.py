"""Database connection, migration, and core CRUD helpers for Super-Memory."""

import datetime
import hashlib
import json
from typing import Optional

import lancedb

from .config import get_config
from .schema import MemorySchema


config = get_config()
db = lancedb.connect(config.db_path)


def _migrate_schema_if_needed() -> None:
    """Migrate from 0.1.0 (text, vector) to 0.2.1 schema using LanceDB add_columns.

    Uses schema evolution (add_columns) instead of recreate:
    1. If table doesn't exist: create it
    2. If table exists with old schema: add new columns with defaults
    3. If table exists with new schema: use as-is
    """
    if "memories" not in list(db.list_tables()):
        try:
            db.create_table("memories", schema=MemorySchema)
        except ValueError as e:
            if "already exists" not in str(e).lower():
                raise
        return

    existing_table = db.open_table("memories")
    column_names = [f.name for f in existing_table.schema]

    if "source_type" in column_names:
        return

    existing_table.add_columns(
        {
            "source_type": "CAST('session' AS STRING)",
            "source_path": "CAST(NULL AS STRING)",
            "timestamp": "CAST(NULL AS TIMESTAMP)",
            "content_hash": "CAST(NULL AS STRING)",
            "metadata_json": "CAST(NULL AS STRING)",
        }
    )


_migrate_schema_if_needed()

table = db.open_table("memories")


def _escape_sql(value: str) -> str:
    """Escape single quotes for SQL string literals to prevent injection."""
    return value.replace("'", "''")


def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content.

    Args:
        content: String content to hash.

    Returns:
        Hexadecimal hash string.
    """
    return hashlib.sha256(content.encode()).hexdigest()


def parse_metadata(metadata: Optional[dict]) -> Optional[str]:
    """Serialize metadata dict to JSON string.

    Args:
        metadata: Optional dictionary to serialize.

    Returns:
        JSON string or None if metadata is None.
    """
    if metadata is None:
        return None
    return json.dumps(metadata)


def add_memory(
    text: str,
    source_type: str,
    source_path: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Add a memory entry to the database.

    Args:
        text: Content text to store.
        source_type: Type of source (session, file, web, boomerang).
        source_path: Optional source path or URL.
        metadata: Optional metadata dictionary.
    """
    entry = {
        "text": text,
        "source_type": source_type,
        "source_path": source_path,
        "content_hash": compute_hash(text),
        "metadata_json": parse_metadata(metadata),
        "timestamp": datetime.datetime.now(),
    }
    table.add([entry])


def query_memories(question: str, top_k: int = 3) -> list[MemorySchema]:
    """Search for relevant memories using semantic similarity.

    Args:
        question: Query text to search for.
        top_k: Maximum number of results to return.

    Returns:
        List of matching MemorySchema entries.
    """
    return table.search(question).limit(top_k).to_pydantic(MemorySchema)


def list_memory_sources(source_type: Optional[str] = None) -> list[MemorySchema]:
    """List all memory sources, optionally filtered by source type.

    Args:
        source_type: Optional filter for source type.

    Returns:
        List of MemorySchema entries.
    """
    if source_type:
        escaped_type = _escape_sql(source_type)
        return (
            table.search()
            .where(f"source_type = '{escaped_type}'")
            .to_pydantic(MemorySchema)
        )
    return table.search().to_pydantic(MemorySchema)


def recall_memory_by_path(source_path: str) -> Optional[MemorySchema]:
    """Retrieve memory by exact source path.

    Args:
        source_path: Source path to look up.

    Returns:
        MemorySchema entry if found, None otherwise.
    """
    escaped_path = _escape_sql(source_path)
    results = (
        table.search()
        .where(f"source_path = '{escaped_path}'")
        .limit(1)
        .to_pydantic(MemorySchema)
    )
    return results[0] if results else None


def save_boomerang_context(session_id: str, context: dict) -> None:
    """Save a boomerang context bundle for later recall.

    Args:
        session_id: Unique session identifier.
        context: Context dictionary to store.
    """
    content = json.dumps(context, indent=2)
    add_memory(
        text=content,
        source_type="boomerang",
        source_path=session_id,
        metadata=None,
    )


def get_boomerang_context(session_id: str) -> Optional[MemorySchema]:
    """Retrieve a boomerang context bundle by session ID.

    Args:
        session_id: Session identifier to look up.

    Returns:
        MemorySchema entry if found, None otherwise.
    """
    escaped_session = _escape_sql(session_id)
    results = (
        table.search()
        .where(f"source_type = 'boomerang' AND source_path = '{escaped_session}'")
        .limit(1)
        .to_pydantic(MemorySchema)
    )
    return results[0] if results else None
