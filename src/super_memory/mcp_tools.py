"""MCP tool definitions for Super-Memory."""

import functools
import logging
from typing import Optional

from fastmcp import FastMCP

from .exceptions import SuperMemoryError
from .memory import (
    add_memory,
    get_boomerang_context as _get_boomerang_context,
    list_memory_sources,
    query_memories,
    recall_memory_by_path,
    save_boomerang_context as _save_boomerang_context,
)

logger = logging.getLogger(__name__)


def _mcp_error_handler(func):
    """Decorator to catch SuperMemoryError and return user-friendly strings.

    Unexpected exceptions (not SuperMemoryError) are allowed to propagate
    so FastMCP can handle them appropriately.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SuperMemoryError as e:
            logger.error("MCP tool %s failed: %s", func.__name__, e)
            return f"Error: {e.message}"

    return wrapper


def register_tools(mcp: FastMCP) -> None:
    """Register all Super-Memory MCP tools.

    Args:
        mcp: FastMCP instance to register tools with.
    """

    @mcp.tool()
    @_mcp_error_handler
    def save_to_memory(content: str, metadata: Optional[dict] = None) -> str:
        """Stores a piece of information into your local long-term memory."""
        logger.info("Saving to memory: %s chars", len(content))
        add_memory(
            text=content,
            source_type="session",
            metadata=metadata,
        )
        return "Memory archived on GPU."

    @mcp.tool()
    def save_file_memory(file_path: str) -> str:
        """Reads a file using markitdown and stores its content in memory."""
        import markitdown

        logger.info("Saving file memory: %s", file_path)
        try:
            md = markitdown.MarkItDown()
            result = md.convert(file_path)
            content = result.text_content
        except FileNotFoundError:
            logger.warning("File not found: %s", file_path)
            return f"Error: File not found: {file_path}"
        except PermissionError:
            logger.warning("Permission denied: %s", file_path)
            return f"Error: Permission denied: {file_path}"
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            return f"Error reading file: {e}"

        add_memory(
            text=content,
            source_type="file",
            source_path=file_path,
        )
        return f"File content archived: {file_path}"

    @mcp.tool()
    def save_web_memory(url: str, title: Optional[str] = None) -> str:
        """Fetches a URL using markitdown and stores its content in memory."""
        import markitdown

        logger.info("Saving web memory: %s", url)
        try:
            md = markitdown.MarkItDown()
            result = md.convert(url)
            content = result.text_content
        except FileNotFoundError:
            logger.warning("URL not found: %s", url)
            return f"Error: URL not found: {url}"
        except PermissionError:
            logger.warning("Permission denied for URL: %s", url)
            return f"Error: Permission denied: {url}"
        except Exception as e:
            logger.error("Failed to fetch URL %s: %s", url, e)
            return f"Error fetching URL: {e}"

        metadata = {"title": title} if title else None
        add_memory(
            text=content,
            source_type="web",
            source_path=url,
            metadata=metadata,
        )
        return f"Web content archived: {url}"

    @mcp.tool()
    @_mcp_error_handler
    def list_sources(source_type: Optional[str] = None) -> str:
        """Lists all memory sources, optionally filtered by source type."""
        logger.info("Listing memory sources")
        results = list_memory_sources(source_type)

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
    @_mcp_error_handler
    def recall_source(source_path: str) -> str:
        """Retrieves memory content by exact source path."""
        logger.info("Recalling source: %s", source_path)
        result = recall_memory_by_path(source_path)

        if not result:
            return f"No memory found for source: {source_path}"

        return result.text

    @mcp.tool()
    @_mcp_error_handler
    def save_boomerang_context(session_id: str, context: dict) -> str:
        """Saves a boomerang context bundle for later recall."""
        logger.info("Saving boomerang context: session=%s", session_id)
        _save_boomerang_context(session_id, context)
        return f"Boomerang context saved for session: {session_id}"

    @mcp.tool()
    @_mcp_error_handler
    def get_boomerang_context(session_id: str) -> str:
        """Retrieves a boomerang context bundle by session ID."""
        logger.info("Getting boomerang context: session=%s", session_id)
        result = _get_boomerang_context(session_id)

        if not result:
            return f"No boomerang context found for session: {session_id}"

        return result.text

    @mcp.tool()
    @_mcp_error_handler
    def query_memory(question: str, top_k: int = 3) -> str:
        """Retrieves relevant past memories based on a semantic search."""
        logger.info("Querying memory: %s", question)
        results = query_memories(question, top_k)

        if not results:
            return "No relevant memories found."

        context = "\n---\n".join([r.text for r in results])
        return f"Found these relevant memories:\n\n{context}"
