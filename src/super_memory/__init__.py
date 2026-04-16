import datetime
import hashlib
import json
from typing import Optional

import lancedb
import torch
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from fastmcp import FastMCP

__version__ = "0.2.1"

mcp = FastMCP("SuperMemory")

DB_PATH = "./memory_data"
db = lancedb.connect(DB_PATH)

registry = get_registry().get("sentence-transformers")
embed_model = registry.create(
    name="BAAI/bge-large-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu"
)


class MemorySchema(LanceModel):
    text: str = embed_model.SourceField()
    vector: Vector(embed_model.ndims()) = embed_model.VectorField()  # type: ignore
    source_type: str = "session"
    source_path: Optional[str] = None
    timestamp: datetime.datetime
    content_hash: Optional[str] = None
    metadata_json: Optional[str] = None


def _migrate_schema_if_needed():
    """Migrate from 0.1.0 (text, vector) to 0.2.1 schema using LanceDB add_columns.

    Uses schema evolution (add_columns) instead of recreate:
    1. If table doesn't exist: create it
    2. If table exists with old schema: add new columns with defaults
    3. If table exists with new schema: use as-is
    """
    if "memories" not in db.table_names():
        db.create_table("memories", schema=MemorySchema)
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


def _compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _parse_metadata(metadata: Optional[dict]) -> Optional[str]:
    if metadata is None:
        return None
    return json.dumps(metadata)


@mcp.tool()
def save_to_memory(content: str, metadata: Optional[dict] = None) -> str:
    """Stores a piece of information into your local long-term memory."""
    entry = {
        "text": content,
        "source_type": "session",
        "content_hash": _compute_hash(content),
        "metadata_json": _parse_metadata(metadata),
        "timestamp": datetime.datetime.now(),
    }
    table.add([entry])
    return "Memory archived on GPU."


@mcp.tool()
def save_file_memory(file_path: str) -> str:
    """Reads a file using markitdown and stores its content in memory."""
    import markitdown

    try:
        md = markitdown.MarkItDown()
        result = md.convert(file_path)
        content = result.text_content
    except Exception as e:
        return f"Error reading file: {e}"

    entry = {
        "text": content,
        "source_type": "file",
        "source_path": file_path,
        "content_hash": _compute_hash(content),
        "timestamp": datetime.datetime.now(),
    }
    table.add([entry])
    return f"File content archived: {file_path}"


@mcp.tool()
def save_web_memory(url: str, title: Optional[str] = None) -> str:
    """Fetches a URL using markitdown and stores its content in memory."""
    import markitdown

    try:
        md = markitdown.MarkItDown()
        result = md.convert(url)
        content = result.text_content
    except Exception as e:
        return f"Error fetching URL: {e}"

    metadata = {"title": title} if title else None
    entry = {
        "text": content,
        "source_type": "web",
        "source_path": url,
        "content_hash": _compute_hash(content),
        "metadata_json": _parse_metadata(metadata),
        "timestamp": datetime.datetime.now(),
    }
    table.add([entry])
    return f"Web content archived: {url}"


@mcp.tool()
def list_sources(source_type: Optional[str] = None) -> str:
    """Lists all memory sources, optionally filtered by source type."""
    if source_type:
        results = (
            table.search()
            .where(f"source_type = '{source_type}'")
            .to_pydantic(MemorySchema)
        )
    else:
        results = table.search().to_pydantic(MemorySchema)

    if not results:
        return "No sources found."

    sources = []
    for r in results:
        if r.source_path:
            sources.append(f"[{r.source_type}] {r.source_path}")
        else:
            sources.append(f"[{r.source_type}] (no path)")

    return "Captured sources:\n\n" + "\n".join(f"- {s}" for s in sources)


@mcp.tool()
def recall_source(source_path: str) -> str:
    """Retrieves memory content by exact source path."""
    results = (
        table.search()
        .where(f"source_path = '{source_path}'")
        .limit(1)
        .to_pydantic(MemorySchema)
    )

    if not results:
        return f"No memory found for source: {source_path}"

    return results[0].text


@mcp.tool()
def save_boomerang_context(session_id: str, context: dict) -> str:
    """Saves a boomerang context bundle for later recall."""
    content = json.dumps(context, indent=2)
    entry = {
        "text": content,
        "source_type": "boomerang",
        "source_path": session_id,
        "content_hash": _compute_hash(content),
        "timestamp": datetime.datetime.now(),
    }
    table.add([entry])
    return f"Boomerang context saved for session: {session_id}"


@mcp.tool()
def get_boomerang_context(session_id: str) -> str:
    """Retrieves a boomerang context bundle by session ID."""
    results = (
        table.search()
        .where(f"source_type = 'boomerang' AND source_path = '{session_id}'")
        .limit(1)
        .to_pydantic(MemorySchema)
    )

    if not results:
        return f"No boomerang context found for session: {session_id}"

    return results[0].text


@mcp.tool()
def query_memory(question: str, top_k: int = 3) -> str:
    """Retrieves relevant past memories based on a semantic search."""
    results = table.search(question).limit(top_k).to_pydantic(MemorySchema)

    if not results:
        return "No relevant memories found."

    context = "\n---\n".join([r.text for r in results])
    return f"Found these relevant memories:\n\n{context}"


def main():
    mcp.run()
