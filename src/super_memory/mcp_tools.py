"""MCP tool definitions for Super-Memory."""

from typing import Optional

from fastmcp import FastMCP

from .memory import (
    add_memory,
    get_boomerang_context as _get_boomerang_context,
    list_memory_sources,
    query_memories,
    recall_memory_by_path,
    save_boomerang_context as _save_boomerang_context,
)


def register_tools(mcp: FastMCP) -> None:
    """Register all Super-Memory MCP tools.

    Args:
        mcp: FastMCP instance to register tools with.
    """

    @mcp.tool()
    def save_to_memory(content: str, metadata: Optional[dict] = None) -> str:
        """Stores a piece of information into your local long-term memory."""
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

        try:
            md = markitdown.MarkItDown()
            result = md.convert(file_path)
            content = result.text_content
        except Exception as e:
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

        try:
            md = markitdown.MarkItDown()
            result = md.convert(url)
            content = result.text_content
        except Exception as e:
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
    def list_sources(source_type: Optional[str] = None) -> str:
        """Lists all memory sources, optionally filtered by source type."""
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
    def recall_source(source_path: str) -> str:
        """Retrieves memory content by exact source path."""
        result = recall_memory_by_path(source_path)

        if not result:
            return f"No memory found for source: {source_path}"

        return result.text

    @mcp.tool()
    def save_boomerang_context(session_id: str, context: dict) -> str:
        """Saves a boomerang context bundle for later recall."""
        _save_boomerang_context(session_id, context)
        return f"Boomerang context saved for session: {session_id}"

    @mcp.tool()
    def get_boomerang_context(session_id: str) -> str:
        """Retrieves a boomerang context bundle by session ID."""
        result = _get_boomerang_context(session_id)

        if not result:
            return f"No boomerang context found for session: {session_id}"

        return result.text

    @mcp.tool()
    def query_memory(question: str, top_k: int = 3) -> str:
        """Retrieves relevant past memories based on a semantic search."""
        results = query_memories(question, top_k)

        if not results:
            return "No relevant memories found."

        context = "\n---\n".join([r.text for r in results])
        return f"Found these relevant memories:\n\n{context}"
