"""Super-Memory: A semantic memory storage and retrieval system."""

from fastmcp import FastMCP

from .mcp_tools import register_tools


__version__ = "0.2.2"

mcp = FastMCP("SuperMemory")

register_tools(mcp)


def main() -> None:
    """Run the Super-Memory MCP server."""
    mcp.run()


__all__ = ["mcp", "main", "__version__"]
