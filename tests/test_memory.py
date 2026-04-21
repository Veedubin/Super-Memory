"""Tests for memory module."""

from __future__ import annotations


# Import functions directly to test
from super_memory.memory import (
    _escape_sql,
    compute_hash,
    parse_metadata,
)


class TestComputeHash:
    """Tests for compute_hash() function."""

    def test_compute_hash_produces_stable_sha256(self) -> None:
        """Test that compute_hash produces consistent SHA256 output."""
        content = "Hello, World!"
        hash1 = compute_hash(content)
        hash2 = compute_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        assert hash1.isalnum()  # Only hex characters

    def test_compute_hash_different_for_different_content(self) -> None:
        """Test that different content produces different hashes."""
        hash1 = compute_hash("content1")
        hash2 = compute_hash("content2")

        assert hash1 != hash2

    def test_compute_hash_empty_string(self) -> None:
        """Test hash of empty string."""
        result = compute_hash("")
        assert len(result) == 64


class TestParseMetadata:
    """Tests for parse_metadata() function."""

    def test_parse_metadata_none_returns_none(self) -> None:
        """Test that None input returns None."""
        result = parse_metadata(None)
        assert result is None

    def test_parse_metadata_dict_returns_json(self) -> None:
        """Test that dict input returns JSON string."""
        metadata = {"key": "value", "number": 42}
        result = parse_metadata(metadata)

        assert result is not None
        assert '"key"' in result
        assert '"value"' in result

    def test_parse_metadata_empty_dict(self) -> None:
        """Test that empty dict returns valid JSON."""
        result = parse_metadata({})
        assert result == "{}"


class TestMemoryOperations:
    """Tests for memory CRUD operations using temporary database."""

    def test_add_and_query_memories_roundtrip(self, memory_db) -> None:
        """Test add_memory and query_memories work together."""
        from super_memory.memory import add_memory, query_memories

        # Add a memory
        add_memory(
            text="Python is a great programming language.",
            source_type="session",
        )

        # Query should find it
        results = query_memories("programming", top_k=5)
        assert len(results) >= 1
        assert "Python" in results[0].text

    def test_list_memory_sources_without_filter(self, memory_db) -> None:
        """Test list_memory_sources returns all sources."""
        from super_memory.memory import add_memory, list_memory_sources

        add_memory(text="Source 1", source_type="session")
        add_memory(text="Source 2", source_type="file")

        results = list_memory_sources()
        assert len(results) >= 2

    def test_list_memory_sources_with_filter(self, memory_db) -> None:
        """Test list_memory_sources with source_type filter."""
        from super_memory.memory import add_memory, list_memory_sources

        add_memory(text="Session source", source_type="session")
        add_memory(text="File source", source_type="file")

        session_sources = list_memory_sources(source_type="session")
        assert all(r.source_type == "session" for r in session_sources)

    def test_recall_memory_by_path(self, memory_db) -> None:
        """Test recall_memory_by_path returns correct memory."""
        from super_memory.memory import add_memory, recall_memory_by_path

        path = "/path/to/test/file.txt"
        add_memory(
            text="File content here",
            source_type="file",
            source_path=path,
        )

        result = recall_memory_by_path(path)
        assert result is not None
        assert result.text == "File content here"
        assert result.source_path == path

    def test_recall_memory_by_path_not_found(self, memory_db) -> None:
        """Test recall_memory_by_path returns None for non-existent path."""
        from super_memory.memory import recall_memory_by_path

        result = recall_memory_by_path("/nonexistent/path.txt")
        assert result is None

    def test_save_and_get_boomerang_context_roundtrip(self, memory_db) -> None:
        """Test save_boomerang_context and get_boomerang_context work together."""
        from super_memory.memory import get_boomerang_context, save_boomerang_context

        session_id = "test-session-123"
        context = {"tasks": ["task1", "task2"], "mode": "testing"}

        save_boomerang_context(session_id, context)

        result = get_boomerang_context(session_id)
        assert result is not None
        assert "task1" in result.text
        assert "task2" in result.text

    def test_get_boomerang_context_not_found(self, memory_db) -> None:
        """Test get_boomerang_context returns None for non-existent session."""
        from super_memory.memory import get_boomerang_context

        result = get_boomerang_context("nonexistent-session")
        assert result is None


class TestSQLInjectionResistance:
    """Tests for SQL injection resistance in path queries."""

    def test_source_path_with_single_quotes(self, memory_db) -> None:
        """Test that single quotes in source_path are properly escaped."""
        from super_memory.memory import add_memory, recall_memory_by_path

        # Attempt SQL injection via single quotes
        malicious_path = "'; DROP TABLE memories; --"
        add_memory(
            text="Memory with special path",
            source_type="session",
            source_path=malicious_path,
        )

        # Should still work and not cause SQL errors
        result = recall_memory_by_path(malicious_path)
        assert result is not None
        assert result.text == "Memory with special path"

    def test_escape_sql_doubles_single_quotes(self) -> None:
        """Test that _escape_sql properly doubles single quotes for SQL safety."""
        result = _escape_sql("test'value")
        # Single quote should be escaped to two single quotes
        assert result == "test''value"
        # The doubled quotes should NOT allow SQL injection
        # In SQL, ' becomes '' which is treated as literal quote
        assert "''" in result
