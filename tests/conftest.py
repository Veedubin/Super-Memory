"""Shared test fixtures for Super-Memory test suite."""

import pytest


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def temp_db_config(temp_db_path):
    """Create a test Config with temporary DB path."""
    import super_memory.config as config_module

    return config_module.Config(
        db_path=temp_db_path,
        device="cpu",
        model="sentence-transformers/all-MiniLM-L6-v2",
        dtype="float32",
        embedding_strategy="TIERED",
        bge_threshold=0.72,
        bge_model="BAAI/bge-large-en-v1.5",
        auto_summarize_interval=15,
    )


@pytest.fixture
def reset_config_cache():
    """Clear config cache before and after test."""
    import super_memory.config as config_module

    config_module.get_config.cache_clear()
    yield
    config_module.get_config.cache_clear()


@pytest.fixture
def memory_db(temp_db_config, monkeypatch, reset_config_cache):
    """Set up an isolated memory database with test config.

    Yields the DB path. Cleans up module state after test.
    """
    import super_memory.config as config_module
    import super_memory.memory as memory_module
    import super_memory.schema as schema_module

    # Store original
    original_get_config = config_module.get_config

    # Monkeypatch config
    monkeypatch.setattr(config_module, "get_config", lambda: temp_db_config)
    monkeypatch.setattr(memory_module, "config", temp_db_config)
    monkeypatch.setattr(schema_module, "config", temp_db_config)

    # Clear lazy-initialized state
    memory_module._db = None
    memory_module._table = None
    schema_module._embed_model = None

    # Reload memory module to pick up new config
    import importlib

    importlib.reload(memory_module)
    importlib.reload(schema_module)

    yield temp_db_config.db_path

    # Restore (fixture cleanup)
    monkeypatch.setattr(config_module, "get_config", original_get_config)
