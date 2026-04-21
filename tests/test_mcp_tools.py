"""Tests for MCP tools module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def get_tool_result_text(result):
    """Extract text content from ToolResult."""
    if hasattr(result, "content") and result.content:
        return (
            result.content[0].text
            if hasattr(result.content[0], "text")
            else str(result.content[0])
        )
    return str(result)


class TestMcpTools:
    """Tests for MCP tools - verifying tool registration and basic behavior."""

    @pytest.fixture
    def setup_mcp(self, memory_db):
        """Set up FastMCP with tools registered."""
        from fastmcp import FastMCP
        from super_memory.mcp_tools import register_tools

        mcp = FastMCP("test")
        register_tools(mcp)
        return mcp

    @pytest.mark.asyncio
    async def test_tools_are_registered(self, setup_mcp) -> None:
        """Test that all 8 expected tools are registered."""
        expected_tools = {
            "save_to_memory",
            "save_file_memory",
            "save_web_memory",
            "list_sources",
            "recall_source",
            "save_boomerang_context",
            "get_boomerang_context",
            "query_memory",
        }

        # Get tool names
        tool_names = set()
        for name in expected_tools:
            tool = await setup_mcp.get_tool(name)
            if tool is not None:
                tool_names.add(name)

        # All tools should be registered
        assert expected_tools.issubset(tool_names), (
            f"Missing tools: {expected_tools - tool_names}"
        )

    @pytest.mark.asyncio
    async def test_save_to_memory_calls_add_memory(self, setup_mcp) -> None:
        """Test that save_to_memory calls add_memory correctly."""
        with patch("super_memory.mcp_tools.add_memory") as mock_add:
            await setup_mcp.call_tool(
                "save_to_memory",
                {"content": "Test content", "metadata": {"key": "value"}},
            )
            mock_add.assert_called_once_with(
                text="Test content",
                source_type="session",
                metadata={"key": "value"},
            )

    @pytest.mark.asyncio
    async def test_query_memory_returns_formatted_results(self, setup_mcp) -> None:
        """Test that query_memory returns formatted results."""
        from super_memory.memory import add_memory

        # Add test data
        add_memory(text="Python is a programming language", source_type="session")

        result = await setup_mcp.call_tool(
            "query_memory", {"question": "programming", "top_k": 3}
        )
        text = get_tool_result_text(result)

        assert "Found these relevant memories:" in text
        assert "Python" in text

    @pytest.mark.asyncio
    async def test_list_sources_returns_sources_list(self, setup_mcp) -> None:
        """Test that list_sources returns formatted sources."""
        from super_memory.memory import add_memory

        add_memory(text="Source 1", source_type="session")
        add_memory(text="Source 2", source_type="file")

        result = await setup_mcp.call_tool("list_sources", {"source_type": None})
        text = get_tool_result_text(result)

        assert "session" in text

    @pytest.mark.asyncio
    async def test_recall_source_returns_content(self, setup_mcp) -> None:
        """Test that recall_source returns memory content."""
        from super_memory.memory import add_memory

        path = "/test/path.txt"
        add_memory(text="Recall content here", source_type="file", source_path=path)

        result = await setup_mcp.call_tool("recall_source", {"source_path": path})
        text = get_tool_result_text(result)

        assert "Recall content here" in text

    @pytest.mark.asyncio
    async def test_recall_source_not_found(self, setup_mcp) -> None:
        """Test recall_source returns message when not found."""
        result = await setup_mcp.call_tool(
            "recall_source", {"source_path": "/nonexistent/path"}
        )
        text = get_tool_result_text(result)

        assert "No memory found" in text

    @pytest.mark.asyncio
    async def test_save_boomerang_context_returns_success(self, setup_mcp) -> None:
        """Test save_boomerang_context returns success message."""
        result = await setup_mcp.call_tool(
            "save_boomerang_context",
            {"session_id": "test-session", "context": {"task": "testing"}},
        )
        text = get_tool_result_text(result)

        assert "saved" in text
        assert "test-session" in text

    @pytest.mark.asyncio
    async def test_get_boomerang_context_returns_content(self, setup_mcp) -> None:
        """Test get_boomerang_context returns stored context."""
        from super_memory.memory import save_boomerang_context

        save_boomerang_context("test-session", {"task": "testing"})

        result = await setup_mcp.call_tool(
            "get_boomerang_context", {"session_id": "test-session"}
        )
        text = get_tool_result_text(result)

        assert "testing" in text

    @pytest.mark.asyncio
    async def test_get_boomerang_context_not_found(self, setup_mcp) -> None:
        """Test get_boomerang_context returns message when not found."""
        result = await setup_mcp.call_tool(
            "get_boomerang_context", {"session_id": "nonexistent"}
        )
        text = get_tool_result_text(result)

        assert "No boomerang context found" in text

    @pytest.mark.asyncio
    async def test_save_file_memory_uses_markitdown(self, setup_mcp, tmp_path) -> None:
        """Test save_file_memory uses MarkItDown."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("File content here")

        mock_result = MagicMock()
        mock_result.text_content = "File content here"

        with patch("markitdown.MarkItDown") as mock_md:
            mock_instance = MagicMock()
            mock_instance.convert.return_value = mock_result
            mock_md.return_value = mock_instance

            result = await setup_mcp.call_tool(
                "save_file_memory", {"file_path": str(test_file)}
            )
            text = get_tool_result_text(result)

            mock_md.assert_called_once()
            assert "test.txt" in text
            assert "archived" in text

    @pytest.mark.asyncio
    async def test_save_web_memory_uses_markitdown(self, setup_mcp) -> None:
        """Test save_web_memory uses MarkItDown."""
        mock_result = MagicMock()
        mock_result.text_content = "Web content"

        with patch("markitdown.MarkItDown") as mock_md:
            mock_instance = MagicMock()
            mock_instance.convert.return_value = mock_result
            mock_md.return_value = mock_instance

            result = await setup_mcp.call_tool(
                "save_web_memory",
                {"url": "https://example.com", "title": "Test Page"},
            )
            text = get_tool_result_text(result)

            mock_md.assert_called_once()
            assert "example.com" in text
            assert "archived" in text
