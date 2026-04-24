"""MCP tool definitions for Super-Memory."""

import functools
import logging
from typing import Optional

from fastmcp import FastMCP

from .exceptions import SuperMemoryError
from .memory import (
    add_memory,
    add_memory_long,
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
    def save_to_memory(
        content: str, metadata: Optional[dict] = None, project: Optional[str] = None
    ) -> str:
        """Stores a piece of information into your local long-term memory.

        Args:
            content: The text content to store.
            metadata: Optional metadata dictionary.
            project: Optional project tag to organize memories.
        """
        logger.info("Saving to memory: %s chars", len(content))
        if project:
            metadata = metadata or {}
            metadata["project"] = project
        add_memory(
            text=content,
            source_type="session",
            metadata=metadata,
        )
        return "Memory archived on GPU."

    @mcp.tool()
    @_mcp_error_handler
    def save_memory_long(
        content: str,
        project: str,
        metadata: Optional[dict] = None,
        force_high_precision: bool = True,
    ) -> str:
        """Stores high-precision memory using BGE-Large embeddings.

        Use this for important content that needs more accurate semantic matching.

        Args:
            content: The text content to store.
            project: Project tag to organize memories.
            metadata: Optional metadata dictionary.
            force_high_precision: If True, always store in high-precision table (default True).
        """
        logger.info(
            "Saving to memory long: %s chars, project=%s", len(content), project
        )
        add_memory_long(
            text=content,
            source_type="session",
            project=project,
            metadata=metadata,
            force_high_precision=force_high_precision,
        )
        return "Memory archived with high precision."

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
    def query_memory(
        question: str, top_k: int = 5, strategy: Optional[str] = None
    ) -> str:
        """Retrieves relevant past memories based on a semantic search.

        Args:
            question: Query text to search for.
            top_k: Maximum number of results to return (default 5, max 20).
            strategy: Optional search strategy override. Options:
                - "tiered": Search MiniLM first, fallback to BGE if low confidence (default)
                - "parallel": Search both tables and merge using rank fusion
                - "minilm_only": Search only MiniLM table (fastest)
        """
        logger.info("Querying memory: %s", question)

        # Normalize strategy
        if strategy is not None:
            strategy = strategy.upper().replace("-", "_")
            if strategy == "MINILM_ONLY":
                strategy = "MINILM_ONLY"
            elif strategy == "PARALLEL":
                strategy = "PARALLEL"
            elif strategy in ("TIERED", "TIERED"):
                strategy = "TIERED"
            else:
                strategy = "TIERED"

        results = query_memories(question, top_k, strategy)

        if not results:
            return "No relevant memories found."

        context = "\n---\n".join([r.text for r in results])
        return f"Found these relevant memories:\n\n{context}"

    @mcp.tool()
    @_mcp_error_handler
    def boomerang_memory_search_tiered(question: str, top_k: int = 5) -> str:
        """Fast tiered memory search - Quick MiniLM search with BGE fallback.

        Use this for routine queries where speed matters.
        This is the "Fast Reply" mode - prioritizes MiniLM speed but falls back
        to BGE-Large if confidence is low (below BGE_THRESHOLD config).

        Args:
            question: Query text to search for.
            top_k: Maximum number of results to return (default 5, max 20).
        """
        logger.info("Tiered memory search: %s", question)
        results = query_memories(question=question, top_k=top_k, strategy_override="TIERED")

        if not results:
            return "No relevant memories found in tiered search."

        context = "\n---\n".join([r.text for r in results])
        return f"[TIERED] Found these relevant memories:\n\n{context}"

    @mcp.tool()
    @_mcp_error_handler
    def boomerang_memory_search_parallel(question: str, top_k: int = 5) -> str:
        """Archivist parallel memory search - Dual-tier search with RRF fusion.

        Use this for important queries where maximum recall matters.
        This is the "Archivist" mode - searches both MiniLM and BGE-Large tables
        simultaneously and merges results using Reciprocal Rank Fusion (RRF).

        Args:
            question: Query text to search for.
            top_k: Maximum number of results to return (default 5, max 20).
        """
        logger.info("Parallel memory search: %s", question)
        results = query_memories(question=question, top_k=top_k, strategy_override="PARALLEL")

        if not results:
            return "No relevant memories found in parallel search."

        context = "\n---\n".join([r.text for r in results])
        return f"[PARALLEL] Found these relevant memories:\n\n{context}"
