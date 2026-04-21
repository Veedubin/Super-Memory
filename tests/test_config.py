"""Tests for config module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from super_memory.config import Config, get_config


class TestGetConfig:
    """Tests for get_config() function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        get_config.cache_clear()

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        get_config.cache_clear()

    def test_get_config_returns_correct_defaults(self) -> None:
        """Test that get_config() returns correct default values."""
        # Clear any env vars that might be set
        with patch.dict(os.environ, {}, clear=True):
            get_config.cache_clear()
            config = get_config()

        assert isinstance(config, Config)
        assert config.db_path == "./memory_data"
        assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.dtype == "float32"
        # Device depends on torch.cuda.is_available(), so just check it's valid
        assert config.device in ("cpu", "cuda")

    def test_env_var_overrides(self) -> None:
        """Test environment variable overrides for all settings."""
        env_vars = {
            "SUPER_MEMORY_DB_PATH": "/custom/db/path",
            "SUPER_MEMORY_MODEL": "custom/model",
            "SUPER_MEMORY_DEVICE": "cpu",
            "SUPER_MEMORY_DTYPE": "float16",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            get_config.cache_clear()
            config = get_config()

        assert config.db_path == "/custom/db/path"
        assert config.model == "custom/model"
        assert config.device == "cpu"
        assert config.dtype == "float16"

    def test_invalid_device_raises_value_error(self) -> None:
        """Test that invalid SUPER_MEMORY_DEVICE raises ValueError."""
        with patch.dict(os.environ, {"SUPER_MEMORY_DEVICE": "invalid_device"}):
            get_config.cache_clear()
            with pytest.raises(ValueError, match="Invalid SUPER_MEMORY_DEVICE"):
                get_config()

    def test_auto_device_resolves_to_cuda_when_available(self) -> None:
        """Test that 'auto' device resolves to cuda when GPU is available."""
        with patch.dict(os.environ, {"SUPER_MEMORY_DEVICE": "auto"}):
            get_config.cache_clear()
            with patch(
                "super_memory.config.torch.cuda.is_available", return_value=True
            ):
                config = get_config()
        assert config.device == "cuda"

    def test_auto_device_resolves_to_cpu_when_no_cuda(self) -> None:
        """Test that 'auto' device resolves to cpu when no GPU."""
        with patch.dict(os.environ, {"SUPER_MEMORY_DEVICE": "auto"}):
            get_config.cache_clear()
            with patch(
                "super_memory.config.torch.cuda.is_available", return_value=False
            ):
                config = get_config()
        assert config.device == "cpu"

    def test_cache_clear_resets_state(self) -> None:
        """Test that cache_clear() properly resets cached config."""
        # Set custom env
        with patch.dict(os.environ, {"SUPER_MEMORY_DB_PATH": "/first/path"}):
            get_config.cache_clear()
            config1 = get_config()
            assert config1.db_path == "/first/path"

        # Clear and set different env
        get_config.cache_clear()
        with patch.dict(os.environ, {"SUPER_MEMORY_DB_PATH": "/second/path"}):
            get_config.cache_clear()
            config2 = get_config()
            assert config2.db_path == "/second/path"
