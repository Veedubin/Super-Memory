"""Tests for schema migration."""

from __future__ import annotations

import pytest


class TestSchemaMigration:
    """Tests for _migrate_schema_if_needed() function."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        db_path = str(tmp_path / "test_migration.db")
        return db_path

    @pytest.fixture
    def setup_memory_db(self, temp_db_path, monkeypatch):
        """Set up a temporary memory database with mocked config."""
        import super_memory.config as config_module
        import super_memory.memory as memory_module

        config_module.get_config.cache_clear()

        test_config = config_module.Config(
            db_path=temp_db_path,
            device="cpu",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

        monkeypatch.setattr(config_module, "get_config", lambda: test_config)
        monkeypatch.setattr(memory_module, "config", test_config)

        import importlib

        importlib.reload(memory_module)

        yield temp_db_path

        # Clean up by restoring original get_config from module
        # The original is still in the module namespace, just reload
        try:
            config_module.get_config.cache_clear()
        except AttributeError:
            # If cache_clear doesn't exist, just pass
            pass

    def test_migrate_schema_creates_new_table(self, temp_db_path, monkeypatch) -> None:
        """Test that new database gets full schema."""
        import super_memory.config as config_module
        import super_memory.memory as memory_module

        config_module.get_config.cache_clear()

        test_config = config_module.Config(
            db_path=temp_db_path,
            device="cpu",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

        monkeypatch.setattr(config_module, "get_config", lambda: test_config)
        monkeypatch.setattr(memory_module, "config", test_config)

        import importlib

        importlib.reload(memory_module)

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

    def test_add_memory_works_with_full_schema(self, setup_memory_db) -> None:
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
