"""Super-Memory: A semantic memory storage and retrieval system."""

from fastmcp import FastMCP

from .mcp_tools import register_tools
from .config import configure_logging


__version__ = "0.4.1"

mcp = FastMCP("SuperMemory")

register_tools(mcp)

configure_logging()


def main() -> None:
    """Run the Super-Memory MCP server."""
    mcp.run()


__all__ = ["mcp", "main", "__version__"]
