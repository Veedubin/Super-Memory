"""Database connection, migration, and core CRUD helpers for Super-Memory."""

import datetime
import hashlib
import json
import logging
from typing import Any, Optional

import lancedb

from .config import get_config
from .exceptions import (
    DatabaseError,
    MigrationError,
    QueryError,
    ValidationError,
)
from .schema import _get_memory_schema

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 1_000_000  # 1MB
_VALID_SOURCE_TYPES = {"session", "file", "web", "boomerang"}


def _validate_source_type(source_type: str) -> None:
    if source_type not in _VALID_SOURCE_TYPES:
        raise ValidationError(
            f"Invalid source_type '{source_type}'. "
            f"Must be one of: {_VALID_SOURCE_TYPES}"
        )


def _validate_source_path(source_path: Optional[str]) -> None:
    if source_path is None:
        return
    # Block only the most dangerous SQL injection patterns that can't be
    # safely escaped. Others (like ; and --) are valid in paths/URLs and
    # are safely handled by _escape_sql.
    forbidden = {"/*", "*/", "xp_", "sp_"}
    lower = source_path.lower()
    for pattern in forbidden:
        if pattern in lower:
            raise ValidationError(
                f"source_path contains forbidden pattern: {pattern}",
                details={"source_path": source_path},
            )


def _validate_text(text: str) -> None:
    if not text or not isinstance(text, str):
        raise ValidationError("text is required and must be a non-empty string")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"text exceeds maximum length of {MAX_TEXT_LENGTH} characters",
            details={"length": len(text), "max": MAX_TEXT_LENGTH},
        )


config = get_config()

# Lazy initialization to avoid import-time crashes
_db = None

# For backward compatibility, expose table via module-level lazy access
_table = None


def get_db():
    """Lazily initialize and return the database connection.

    Returns:
        LanceDB database connection.

    Raises:
        DatabaseError: If database connection fails.
    """
    global _db
    if _db is None:
        try:
            _db = lancedb.connect(config.db_path)
        except Exception as e:
            logger.error("Failed to connect to database: %s", e)
            raise DatabaseError(f"Failed to connect to database: {e}") from e
    return _db


def get_table():
    """Lazily open and return the memories table.

    Returns:
        LanceDB table instance.

    Raises:
        DatabaseError: If table cannot be opened or created.
    """
    global _table
    if _table is None:
        _table = _get_or_create_table()
    return _table


def __getattr__(name):
    """Module-level attribute access for backward compatibility."""
    if name == "table":
        ensure_initialized()
        return get_table()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_or_create_table():
    """Get existing table or create new one with proper error handling.

    Returns:
        LanceDB table instance.

    Raises:
        DatabaseError: If table operations fail.
    """
    db = get_db()
    try:
        if "memories" in db.list_tables():
            return db.open_table("memories")
    except Exception as e:
        logger.error("Failed to list tables: %s", e)
        raise DatabaseError(f"Failed to list tables: {e}") from e

    try:
        return db.create_table("memories", schema=_get_memory_schema())
    except Exception as e:
        # Handle race condition where table was created between check and create
        if "already exists" in str(e).lower():
            try:
                return db.open_table("memories")
            except Exception as open_err:
                logger.error(
                    "Table creation reported success but open failed: %s", open_err
                )
                raise DatabaseError(
                    f"Table creation reported success but open failed: {open_err}"
                ) from open_err
        logger.error("Failed to create table: %s", e)
        raise DatabaseError(f"Failed to create table: {e}") from e


def _validate_vector_dimensions() -> None:
    """Validate that stored vectors match current model dimensions.

    Warns if dimensions mismatch, which can happen when switching models.
    """
    from .schema import get_embed_model

    try:
        table = get_table()
        expected_dims = get_embed_model().ndims()

        # Check schema for vector field dimensions
        schema = table.schema
        for field in schema:
            if field.name == "vector":
                # LanceDB vector field type contains dimension info
                actual_dims = field.type.list_size
                if actual_dims != expected_dims:
                    logger.warning(
                        "Vector dimension mismatch: DB has %d dims, model produces %d dims. "
                        "This will cause search failures. Consider re-creating the database "
                        "or switching back to the original model.",
                        actual_dims,
                        expected_dims,
                    )
                break
    except Exception as e:
        logger.debug("Could not validate vector dimensions: %s", e)


def ensure_initialized():
    """Ensure database and table are initialized. Call before operations that need them."""
    _migrate_schema_if_needed()
    get_table()
    _validate_vector_dimensions()


def _migrate_schema_if_needed() -> None:
    """Migrate from 0.1.0 (text, vector) to 0.2.1 schema using LanceDB add_columns.

    Uses schema evolution (add_columns) instead of recreate:
    1. If table doesn't exist: create it
    2. If table exists with old schema: add new columns with defaults
    3. If table exists with new schema: use as-is
    4. Handle corrupted/missing tables gracefully

    Raises:
        MigrationError: If migration fails.
    """
    db = get_db()

    try:
        existing_tables = db.list_tables()
    except Exception as e:
        logger.error("Failed to list tables (DB may be corrupted): %s", e)
        raise MigrationError(f"Failed to list tables (DB may be corrupted): {e}") from e

    if "memories" not in existing_tables:
        try:
            db.create_table("memories", schema=_get_memory_schema())
        except ValueError as e:
            if "already exists" not in str(e).lower():
                logger.error("Failed to create memories table: %s", e)
                raise MigrationError(f"Failed to create memories table: {e}") from e
        except Exception as e:
            logger.error("Failed to create memories table: %s", e)
            raise MigrationError(f"Failed to create memories table: {e}") from e
        return

    try:
        existing_table = db.open_table("memories")
    except Exception as e:
        # Table listed but can't be opened - likely corrupted, recreate it
        if "not found" in str(e).lower() or "lance" in str(e).lower():
            try:
                # Drop corrupted table and recreate
                if hasattr(db, "drop_table"):
                    db.drop_table("memories")
            except Exception:
                pass  # Drop may fail but we still try to create
            try:
                db.create_table("memories", schema=_get_memory_schema())
                return
            except Exception as create_err:
                logger.error(
                    "Table was corrupted and recreation failed: %s", create_err
                )
                raise MigrationError(
                    f"Table was corrupted and recreation failed: {create_err}"
                ) from create_err
        logger.error("Failed to open memories table: %s", e)
        raise MigrationError(f"Failed to open memories table: {e}") from e

    try:
        column_names = [f.name for f in existing_table.schema]
    except Exception as e:
        logger.error("Failed to read table schema: %s", e)
        raise MigrationError(f"Failed to read table schema: {e}") from e

    if "source_type" in column_names:
        return

    try:
        existing_table.add_columns(
            {
                "source_type": "CAST('session' AS STRING)",
                "source_path": "CAST(NULL AS STRING)",
                "timestamp": "CAST(NULL AS TIMESTAMP)",
                "content_hash": "CAST(NULL AS STRING)",
                "metadata_json": "CAST(NULL AS STRING)",
            }
        )
    except Exception as e:
        logger.error("Schema migration failed: %s", e)
        raise MigrationError(f"Schema migration failed: {e}") from e


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

    Raises:
        ValidationError: If input validation fails.
        DatabaseError: If database operation fails.
    """
    _validate_text(text)
    _validate_source_type(source_type)
    _validate_source_path(source_path)

    logger.info(
        "Adding memory: source_type=%s, source_path=%s", source_type, source_path
    )

    ensure_initialized()
    table = get_table()

    entry = {
        "text": text,
        "source_type": source_type,
        "source_path": source_path,
        "content_hash": compute_hash(text),
        "metadata_json": parse_metadata(metadata),
        "timestamp": datetime.datetime.now(),
    }

    try:
        table.add([entry])
    except Exception as e:
        logger.error("Failed to add memory: %s", e)
        raise DatabaseError(f"Failed to add memory: {e}") from e


def add_memories(
    entries: list[dict[str, Any]],
    *,
    atomic: bool = False,
) -> list[dict[str, Any]]:
    """Add multiple memory entries to the database.

    Args:
        entries: List of dicts, each with keys:
            - text (str, required)
            - source_type (str, required)
            - source_path (Optional[str])
            - metadata (Optional[dict])
        atomic: If True, validate all entries before adding any.
            If False, add individually and collect per-entry results.

    Returns:
        List of result dicts with keys:
            - success (bool)
            - hash (Optional[str]): Content hash if successful
            - error (Optional[str]): Error message if failed

    Raises:
        ValidationError: If atomic=True and any entry is invalid.
        DatabaseError: If atomic=True and the batch DB write fails.
    """
    if not entries:
        logger.debug("add_memories called with empty entries list")
        return []

    logger.info("Adding %d memories (atomic=%s)", len(entries), atomic)

    # Preprocess and validate all entries
    processed = []
    for i, entry in enumerate(entries):
        text = entry.get("text")
        _validate_text(text)

        source_type = entry.get("source_type")
        if not source_type or not isinstance(source_type, str):
            raise ValidationError(
                f"Entry {i}: 'source_type' is required and must be a string"
            )
        _validate_source_type(source_type)

        source_path = entry.get("source_path")
        _validate_source_path(source_path)

        processed.append(
            {
                "text": text,
                "source_type": source_type,
                "source_path": source_path,
                "content_hash": compute_hash(text),
                "metadata_json": parse_metadata(entry.get("metadata")),
                "timestamp": datetime.datetime.now(),
            }
        )

    table = get_table()

    if atomic:
        try:
            table.add(processed)
            logger.info("Successfully added %d memories atomically", len(processed))
            return [
                {"success": True, "hash": e["content_hash"], "error": None}
                for e in processed
            ]
        except Exception as e:
            logger.error("Batch add failed: %s", e)
            raise DatabaseError(f"Batch add failed: {e}") from e

    # Non-atomic: add individually
    results = []
    for entry in processed:
        try:
            table.add([entry])
            results.append(
                {"success": True, "hash": entry["content_hash"], "error": None}
            )
        except Exception as e:
            logger.error("Failed to add memory: %s", e)
            results.append({"success": False, "hash": None, "error": str(e)})

    success_count = sum(1 for r in results if r["success"])
    logger.info("Added %d/%d memories individually", success_count, len(results))
    return results


def query_memories(question: str, top_k: int = 5) -> list:
    """Search for relevant memories using semantic similarity.

    Args:
        question: Query text to search for.
        top_k: Maximum number of results to return.

    Returns:
        List of matching MemorySchema entries.

    Raises:
        ValidationError: If top_k is not a positive integer or exceeds 20.
        QueryError: If database query fails.
    """
    if top_k < 1:
        raise ValidationError("top_k must be a positive integer")
    if top_k > 20:
        raise ValidationError("top_k cannot exceed 20")
    logger.info("Querying memories: question=%s, top_k=%d", question, top_k)

    ensure_initialized()
    table = get_table()
    schema = _get_memory_schema()

    try:
        return table.search(question).limit(top_k).to_pydantic(schema)
    except Exception as e:
        logger.error("Failed to query memories: %s", e)
        raise QueryError(f"Failed to query memories: {e}") from e


def list_memory_sources(source_type: Optional[str] = None) -> list:
    """List all memory sources, optionally filtered by source type.

    Args:
        source_type: Optional filter for source type.

    Returns:
        List of MemorySchema entries.

    Raises:
        QueryError: If database query fails.
    """
    logger.info("Listing memory sources: source_type=%s", source_type)

    ensure_initialized()
    table = get_table()
    schema = _get_memory_schema()

    try:
        if source_type:
            escaped_type = _escape_sql(source_type)
            return (
                table.search()
                .where(f"source_type = '{escaped_type}'")
                .to_pydantic(schema)
            )
        return table.search().to_pydantic(schema)
    except Exception as e:
        logger.error("Failed to list memory sources: %s", e)
        raise QueryError(f"Failed to list memory sources: {e}") from e


def recall_memory_by_path(source_path: str) -> Optional:
    """Retrieve memory by exact source path.

    Args:
        source_path: Source path to look up.

    Returns:
        MemorySchema entry if found, None otherwise.

    Raises:
        QueryError: If database query fails.
    """
    logger.info("Recalling memory by path: source_path=%s", source_path)

    ensure_initialized()
    table = get_table()
    schema = _get_memory_schema()

    escaped_path = _escape_sql(source_path)

    try:
        results = (
            table.search()
            .where(f"source_path = '{escaped_path}'")
            .limit(1)
            .to_pydantic(schema)
        )
        return results[0] if results else None
    except Exception as e:
        logger.error("Failed to recall memory: %s", e)
        raise QueryError(f"Failed to recall memory: {e}") from e


def save_boomerang_context(session_id: str, context: dict) -> None:
    """Save a boomerang context bundle for later recall.

    Args:
        session_id: Unique session identifier.
        context: Context dictionary to store.

    Raises:
        ValidationError: If input validation fails.
        DatabaseError: If database operation fails.
    """
    content = json.dumps(context, indent=2)
    add_memory(
        text=content,
        source_type="boomerang",
        source_path=session_id,
        metadata=None,
    )


def get_boomerang_context(session_id: str) -> Optional:
    """Retrieve a boomerang context bundle by session ID.

    Args:
        session_id: Session identifier to look up.

    Returns:
        MemorySchema entry if found, None otherwise.

    Raises:
        QueryError: If database query fails.
    """
    logger.info("Getting boomerang context: session_id=%s", session_id)

    ensure_initialized()
    table = get_table()
    schema = _get_memory_schema()

    escaped_session = _escape_sql(session_id)

    try:
        results = (
            table.search()
            .where(f"source_type = 'boomerang' AND source_path = '{escaped_session}'")
            .limit(1)
            .to_pydantic(schema)
        )
        return results[0] if results else None
    except Exception as e:
        logger.error("Failed to get boomerang context: %s", e)
        raise QueryError(f"Failed to get boomerang context: {e}") from e
