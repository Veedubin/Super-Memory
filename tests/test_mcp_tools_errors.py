"""Tests for MCP tool error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from super_memory.exceptions import QueryError, ValidationError


def get_tool_result_text(result):
    """Extract text content from ToolResult."""
    if hasattr(result, "content") and result.content:
        return (
            result.content[0].text
            if hasattr(result.content[0], "text")
            else str(result.content[0])
        )
    return str(result)


class TestMcpToolErrorHandling:
    """Tests for MCP tool error handling and error message formatting."""

    @pytest.fixture
    def mock_memory_db(self, tmp_path, monkeypatch):
        """Set up a temporary memory database."""
        import super_memory.config as config_module
        import super_memory.memory as memory_module

        # Store original get_config
        original_get_config = config_module.get_config

        config_module.get_config.cache_clear()

        db_path = str(tmp_path / "test_mcp_tools_errors.db")
        test_config = config_module.Config(
            db_path=db_path,
            device="cpu",
            model="sentence-transformers/all-MiniLM-L6-v2",
            dtype="float32",
        )

        monkeypatch.setattr(config_module, "get_config", lambda: test_config)
        monkeypatch.setattr(memory_module, "config", test_config)

        import importlib

        importlib.reload(memory_module)

        yield db_path

        # Clean up - restore original and clear cache
        monkeypatch.setattr(config_module, "get_config", original_get_config)
        config_module.get_config.cache_clear()

    @pytest.fixture
    def setup_mcp(self, mock_memory_db):
        """Set up FastMCP with tools registered."""
        from fastmcp import FastMCP
        from super_memory.mcp_tools import register_tools

        mcp = FastMCP("test")
        register_tools(mcp)
        return mcp

    @pytest.mark.asyncio
    async def test_save_to_memory_returns_error_when_add_memory_raises_validation_error(
        self,
        setup_mcp,
    ) -> None:
        """Test that save_to_memory returns 'Error: ...' when add_memory raises ValidationError."""
        with patch("super_memory.mcp_tools.add_memory") as mock_add_memory:
            mock_add_memory.side_effect = ValidationError(
                "text is required and must be a non-empty string"
            )

            result = await setup_mcp.call_tool(
                "save_to_memory", {"content": "", "metadata": None}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "text is required" in text

    @pytest.mark.asyncio
    async def test_save_to_memory_returns_error_for_invalid_source_type(
        self,
        setup_mcp,
    ) -> None:
        """Test that save_to_memory returns 'Error: ...' for invalid source_type."""
        with patch("super_memory.mcp_tools.add_memory") as mock_add_memory:
            mock_add_memory.side_effect = ValidationError(
                "Invalid source_type 'bad_type'. Must be one of: {'session', 'file', 'web', 'boomerang'}"
            )

            result = await setup_mcp.call_tool(
                "save_to_memory", {"content": "test content", "metadata": None}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "Invalid source_type" in text

    @pytest.mark.asyncio
    async def test_query_memory_returns_error_when_query_memories_raises_query_error(
        self,
        setup_mcp,
    ) -> None:
        """Test that query_memory returns 'Error: ...' when query_memories raises QueryError."""
        with patch("super_memory.mcp_tools.query_memories") as mock_query_memories:
            mock_query_memories.side_effect = QueryError(
                "Failed to query memories: search failed"
            )

            result = await setup_mcp.call_tool(
                "query_memory", {"question": "test query", "top_k": 5}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "Failed to query memories" in text

    @pytest.mark.asyncio
    async def test_query_memory_returns_error_for_database_failure(
        self,
        setup_mcp,
    ) -> None:
        """Test that query_memory returns 'Error: ...' for general database failures."""
        with patch("super_memory.mcp_tools.query_memories") as mock_query_memories:
            mock_query_memories.side_effect = QueryError("Database connection lost")

            result = await setup_mcp.call_tool(
                "query_memory", {"question": "test query", "top_k": 5}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "Database connection lost" in text

    @pytest.mark.asyncio
    async def test_save_file_memory_returns_error_file_not_found(
        self,
        setup_mcp,
    ) -> None:
        """Test that save_file_memory returns 'Error: File not found' when file doesn't exist."""
        nonexistent_path = "/nonexistent/path/to/file.txt"

        with patch("markitdown.MarkItDown") as mock_md:
            mock_instance = MagicMock()
            mock_instance.convert.side_effect = FileNotFoundError(
                f"[Errno 2] No such file or directory: '{nonexistent_path}'"
            )
            mock_md.return_value = mock_instance

            result = await setup_mcp.call_tool(
                "save_file_memory", {"file_path": nonexistent_path}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "File not found" in text

    @pytest.mark.asyncio
    async def test_save_file_memory_returns_error_permission_denied(
        self,
        setup_mcp,
    ) -> None:
        """Test that save_file_memory returns 'Error: Permission denied' when permission is denied."""
        protected_path = "/protected/file.txt"

        with patch("markitdown.MarkItDown") as mock_md:
            mock_instance = MagicMock()
            mock_instance.convert.side_effect = PermissionError(
                f"[Errno 13] Permission denied: '{protected_path}'"
            )
            mock_md.return_value = mock_instance

            result = await setup_mcp.call_tool(
                "save_file_memory", {"file_path": protected_path}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "Permission denied" in text

    @pytest.mark.asyncio
    async def test_save_file_memory_returns_error_for_other_exceptions(
        self,
        setup_mcp,
    ) -> None:
        """Test that save_file_memory returns 'Error reading file: ...' for other exceptions."""
        test_path = "/some/file.txt"

        with patch("markitdown.MarkItDown") as mock_md:
            mock_instance = MagicMock()
            mock_instance.convert.side_effect = RuntimeError("Unexpected error")
            mock_md.return_value = mock_instance

            result = await setup_mcp.call_tool(
                "save_file_memory", {"file_path": test_path}
            )
            text = get_tool_result_text(result)

            assert "Error reading file" in text
            assert "Unexpected error" in text

    @pytest.mark.asyncio
    async def test_save_web_memory_returns_error_url_not_found(
        self,
        setup_mcp,
    ) -> None:
        """Test that save_web_memory returns 'Error: URL not found' for invalid URLs."""
        with patch("markitdown.MarkItDown") as mock_md:
            mock_instance = MagicMock()
            mock_instance.convert.side_effect = FileNotFoundError(
                "URL not found: https://invalid-example.com"
            )
            mock_md.return_value = mock_instance

            result = await setup_mcp.call_tool(
                "save_web_memory", {"url": "https://invalid-example.com"}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "URL not found" in text

    @pytest.mark.asyncio
    async def test_save_web_memory_returns_error_permission_denied(
        self,
        setup_mcp,
    ) -> None:
        """Test that save_web_memory returns 'Error: Permission denied' for permission issues."""
        with patch("markitdown.MarkItDown") as mock_md:
            mock_instance = MagicMock()
            mock_instance.convert.side_effect = PermissionError(
                "Permission denied for URL"
            )
            mock_md.return_value = mock_instance

            result = await setup_mcp.call_tool(
                "save_web_memory", {"url": "https://example.com"}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "Permission denied" in text

    @pytest.mark.asyncio
    async def test_list_sources_returns_error_when_list_memory_sources_fails(
        self,
        setup_mcp,
    ) -> None:
        """Test that list_sources returns 'Error: ...' when list_memory_sources fails."""
        with patch("super_memory.mcp_tools.list_memory_sources") as mock_list_sources:
            mock_list_sources.side_effect = QueryError(
                "Failed to list memory sources: query failed"
            )

            result = await setup_mcp.call_tool("list_sources", {"source_type": None})
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "Failed to list memory sources" in text

    @pytest.mark.asyncio
    async def test_recall_source_returns_error_when_recall_memory_by_path_fails(
        self,
        setup_mcp,
    ) -> None:
        """Test that recall_source returns 'Error: ...' when recall_memory_by_path fails."""
        with patch("super_memory.mcp_tools.recall_memory_by_path") as mock_recall:
            mock_recall.side_effect = QueryError(
                "Failed to recall memory: query failed"
            )

            result = await setup_mcp.call_tool(
                "recall_source", {"source_path": "/some/path"}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "Failed to recall memory" in text

    @pytest.mark.asyncio
    async def test_save_boomerang_context_returns_error_when_add_memory_fails(
        self,
        setup_mcp,
    ) -> None:
        """Test that save_boomerang_context returns 'Error: ...' when add_memory fails."""
        with patch("super_memory.mcp_tools._save_boomerang_context") as mock_save:
            mock_save.side_effect = ValidationError("text exceeds maximum length")

            result = await setup_mcp.call_tool(
                "save_boomerang_context",
                {"session_id": "test-session", "context": {"key": "value"}},
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "text exceeds maximum length" in text

    @pytest.mark.asyncio
    async def test_get_boomerang_context_returns_error_when_get_boomerang_context_fails(
        self,
        setup_mcp,
    ) -> None:
        """Test that get_boomerang_context returns 'Error: ...' when get_boomerang_context fails."""
        with patch("super_memory.mcp_tools._get_boomerang_context") as mock_get:
            mock_get.side_effect = QueryError(
                "Failed to get boomerang context: query failed"
            )

            result = await setup_mcp.call_tool(
                "get_boomerang_context", {"session_id": "test-session"}
            )
            text = get_tool_result_text(result)

            assert text.startswith("Error:")
            assert "Failed to get boomerang context" in text
