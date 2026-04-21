"""Tests for schema migration."""

from __future__ import annotations


class TestSchemaMigration:
    """Tests for _migrate_schema_if_needed() function."""

    def test_migrate_schema_creates_new_table(self, memory_db) -> None:

        from super_memory.memory import table

        # Check that table has all required columns
        column_names = [f.name for f in table.schema]
        assert "text" in column_names
        assert "vector" in column_names
        assert "source_type" in column_names
        assert "source_path" in column_names
        assert "timestamp" in column_names
        assert "content_hash" in column_names
        assert "metadata_json" in column_names

    def test_add_memory_works_with_full_schema(self, memory_db) -> None:
        """Test that add_memory works with full schema columns."""
        from super_memory.memory import add_memory

        add_memory(
            text="Test memory",
            source_type="session",
            source_path="/test/path",
            metadata={"key": "value"},
        )

        from super_memory.memory import recall_memory_by_path

        result = recall_memory_by_path("/test/path")

        assert result is not None
        assert result.text == "Test memory"
        assert result.source_type == "session"
        assert result.metadata_json is not None
