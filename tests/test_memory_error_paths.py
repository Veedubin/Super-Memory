"""Tests for error handling in memory module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from super_memory.exceptions import QueryError, ValidationError
from super_memory.memory import (
    MAX_TEXT_LENGTH,
    add_memory,
    list_memory_sources,
    query_memories,
    recall_memory_by_path,
)


class TestAddMemoryValidationErrors:
    """Tests for ValidationError handling in add_memory."""

    def test_add_memory_raises_validation_error_for_empty_text(self) -> None:
        """Test that add_memory raises ValidationError for empty text."""
        with pytest.raises(ValidationError) as exc_info:
            add_memory(text="", source_type="session")
        assert "text is required" in str(exc_info.value)

    def test_add_memory_raises_validation_error_for_none_text(self) -> None:
        """Test that add_memory raises ValidationError for None text."""
        with pytest.raises(ValidationError) as exc_info:
            add_memory(text=None, source_type="session")  # type: ignore
        assert "text is required" in str(exc_info.value)

    def test_add_memory_raises_validation_error_for_text_exceeding_max_length(
        self,
    ) -> None:
        """Test that add_memory raises ValidationError for text exceeding MAX_TEXT_LENGTH."""
        long_text = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            add_memory(text=long_text, source_type="session")
        assert "exceeds maximum length" in str(exc_info.value)
        assert exc_info.value.details.get("length") == MAX_TEXT_LENGTH + 1
        assert exc_info.value.details.get("max") == MAX_TEXT_LENGTH

    def test_add_memory_raises_validation_error_for_invalid_source_type(self) -> None:
        """Test that add_memory raises ValidationError for invalid source_type."""
        with pytest.raises(ValidationError) as exc_info:
            add_memory(text="test", source_type="invalid_type")
        assert "Invalid source_type" in str(exc_info.value)
        assert "invalid_type" in str(exc_info.value)

    def test_add_memory_raises_validation_error_for_source_path_with_forbidden_pattern(
        self,
    ) -> None:
        """Test that add_memory raises ValidationError for source_path with forbidden patterns."""
        forbidden_paths = [
            "/*comment*/",
            "*/other",
            "xp_something",
            "sp_exec",
        ]
        for path in forbidden_paths:
            with pytest.raises(ValidationError) as exc_info:
                add_memory(text="test", source_type="session", source_path=path)
            assert "forbidden pattern" in str(exc_info.value)


class TestQueryMemoriesErrorHandling:
    """Tests for QueryError handling in query_memories."""

    def test_query_memories_raises_query_error_when_search_fails(
        self,
        memory_db,
    ) -> None:
        """Test that query_memories raises QueryError when search fails."""
        with patch("super_memory.memory.get_table") as mock_get_table:
            mock_table = MagicMock()
            # Make search() return a mock that raises on limit().to_pydantic()
            mock_search = MagicMock()
            mock_search.limit.return_value.to_pydantic.side_effect = Exception(
                "Search failed"
            )
            mock_table.search.return_value = mock_search
            mock_get_table.return_value = mock_table

            with pytest.raises(QueryError) as exc_info:
                query_memories("test query")
            assert "Failed to query memories" in str(exc_info.value)

    def test_query_memories_wraps_exception_as_query_error(
        self,
        memory_db,
    ) -> None:
        """Test that query_memories wraps non-QueryError exceptions as QueryError."""
        with patch("super_memory.memory.get_table") as mock_get_table:
            mock_table = MagicMock()
            mock_search = MagicMock()
            mock_search.limit.return_value.to_pydantic.side_effect = RuntimeError(
                "Unexpected error"
            )
            mock_table.search.return_value = mock_search
            mock_get_table.return_value = mock_table

            with pytest.raises(QueryError) as exc_info:
                query_memories("test query")
            # Original error should be chained
            assert exc_info.value.__cause__ is not None


class TestListMemorySourcesErrorHandling:
    """Tests for QueryError handling in list_memory_sources."""

    def test_list_memory_sources_raises_query_error_when_query_fails(
        self,
        memory_db,
    ) -> None:
        """Test that list_memory_sources raises QueryError when query fails."""
        with patch("super_memory.memory.get_table") as mock_get_table:
            mock_table = MagicMock()
            mock_search = MagicMock()
            mock_search.to_pydantic.side_effect = Exception("Query failed")
            mock_table.search.return_value = mock_search
            mock_get_table.return_value = mock_table

            with pytest.raises(QueryError) as exc_info:
                list_memory_sources()
            assert "Failed to list memory sources" in str(exc_info.value)

    def test_list_memory_sources_raises_query_error_when_query_with_filter_fails(
        self,
        memory_db,
    ) -> None:
        """Test that list_memory_sources raises QueryError when filtered query fails."""
        with patch("super_memory.memory.get_table") as mock_get_table:
            mock_table = MagicMock()
            mock_search = MagicMock()
            mock_search.where.return_value.to_pydantic.side_effect = Exception(
                "Filtered query failed"
            )
            mock_table.search.return_value = mock_search
            mock_get_table.return_value = mock_table

            with pytest.raises(QueryError) as exc_info:
                list_memory_sources(source_type="session")
            assert "Failed to list memory sources" in str(exc_info.value)


class TestRecallMemoryByPathErrorHandling:
    """Tests for QueryError handling in recall_memory_by_path."""

    def test_recall_memory_by_path_raises_query_error_when_query_fails(
        self,
        memory_db,
    ) -> None:
        """Test that recall_memory_by_path raises QueryError when query fails."""
        with patch("super_memory.memory.get_table") as mock_get_table:
            mock_table = MagicMock()
            mock_search = MagicMock()
            mock_search.where.return_value.limit.return_value.to_pydantic.side_effect = Exception(
                "Query failed"
            )
            mock_table.search.return_value = mock_search
            mock_get_table.return_value = mock_table

            with pytest.raises(QueryError) as exc_info:
                recall_memory_by_path("/some/path")
            assert "Failed to recall memory" in str(exc_info.value)

    def test_recall_memory_by_path_wraps_exception_as_query_error(
        self,
        memory_db,
    ) -> None:
        """Test that recall_memory_by_path wraps non-QueryError exceptions as QueryError."""
        with patch("super_memory.memory.get_table") as mock_get_table:
            mock_table = MagicMock()
            mock_search = MagicMock()
            mock_search.where.return_value.limit.return_value.to_pydantic.side_effect = RuntimeError(
                "Unexpected error"
            )
            mock_table.search.return_value = mock_search
            mock_get_table.return_value = mock_table

            with pytest.raises(QueryError) as exc_info:
                recall_memory_by_path("/some/path")
            # Original error should be chained
            assert exc_info.value.__cause__ is not None
