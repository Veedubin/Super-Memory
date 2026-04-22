"""Entry point for super_memory package."""

import sys

from super_memory import main as server_main


def main():
    """Run either the MCP server or the migration CLI.

    When run as `python -m super_memory.migrate`, it runs the migration CLI.
    When run as `python -m super_memory` or via `super-memory-mcp`, it runs the MCP server.
    """
    if len(sys.argv) > 0 and "migrate" in sys.argv[0]:
        # Invoked as python -m super_memory.migrate
        from super_memory.migrate import main as migrate_main

        migrate_main()
    else:
        server_main()


if __name__ == "__main__":
    main()
