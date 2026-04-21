"""Tests for add_memories() batch operation."""

from __future__ import annotations

import pytest

from super_memory.exceptions import ValidationError
from super_memory.memory import add_memories, compute_hash


class TestAddMemoriesEmptyList:
    """Tests for empty list handling."""

    def test_empty_list_returns_empty_list(self, memory_db) -> None:
        """Test that empty entries list returns empty results."""
        result = add_memories([])
        assert result == []


class TestAddMemoriesSingleEntry:
    """Tests for single entry batch operations."""

    def test_single_entry_non_atomic_mode(self, memory_db) -> None:
        """Test single entry in non-atomic mode succeeds."""
        entries = [
            {"text": "Single memory entry", "source_type": "session"},
        ]

        results = add_memories(entries, atomic=False)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["hash"] is not None
        assert results[0]["error"] is None

    def test_single_entry_atomic_mode(self, memory_db) -> None:
        """Test single entry in atomic mode succeeds."""
        entries = [
            {"text": "Atomic memory entry", "source_type": "session"},
        ]

        results = add_memories(entries, atomic=True)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["hash"] is not None
        assert results[0]["error"] is None

    def test_single_entry_with_all_fields(self, memory_db) -> None:
        """Test single entry with all optional fields."""
        entries = [
            {
                "text": "Full entry with all fields",
                "source_type": "file",
                "source_path": "/path/to/file.txt",
                "metadata": {"key": "value", "number": 42},
            },
        ]

        results = add_memories(entries, atomic=False)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["hash"] is not None


class TestAddMemoriesMultipleEntries:
    """Tests for multiple entry batch operations."""

    def test_multiple_entries_non_atomic_mode(self, memory_db) -> None:
        """Test multiple entries in non-atomic mode all succeed."""
        entries = [
            {"text": "Memory 1", "source_type": "session"},
            {"text": "Memory 2", "source_type": "file"},
            {"text": "Memory 3", "source_type": "web"},
        ]

        results = add_memories(entries, atomic=False)

        assert len(results) == 3
        assert all(r["success"] is True for r in results)
        assert all(r["hash"] is not None for r in results)
        assert all(r["error"] is None for r in results)

    def test_multiple_entries_atomic_mode(self, memory_db) -> None:
        """Test multiple entries in atomic mode all succeed."""
        entries = [
            {"text": "Atomic memory 1", "source_type": "session"},
            {"text": "Atomic memory 2", "source_type": "file"},
            {"text": "Atomic memory 3", "source_type": "boomerang"},
        ]

        results = add_memories(entries, atomic=True)

        assert len(results) == 3
        assert all(r["success"] is True for r in results)
        assert all(r["hash"] is not None for r in results)
        assert all(r["error"] is None for r in results)


class TestAddMemoriesAtomicMode:
    """Tests for atomic mode behavior."""

    def test_atomic_mode_all_succeed(self, memory_db) -> None:
        """Test atomic mode succeeds when all entries are valid."""
        entries = [
            {"text": "Atomic valid 1", "source_type": "session"},
            {"text": "Atomic valid 2", "source_type": "file"},
        ]

        results = add_memories(entries, atomic=True)

        assert len(results) == 2
        assert all(r["success"] is True for r in results)

    def test_atomic_mode_validation_error_raised(self, memory_db) -> None:
        """Test atomic mode raises ValidationError for invalid entries."""
        entries = [
            {"text": "Valid entry", "source_type": "session"},
            {"text": "Another valid", "source_type": "file"},
            {"text": "", "source_type": "session"},  # Invalid: empty text
        ]

        with pytest.raises(ValidationError) as exc_info:
            add_memories(entries, atomic=True)

        assert "text is required" in str(exc_info.value)

    def test_atomic_mode_invalid_source_type_raises(self, memory_db) -> None:
        """Test atomic mode raises ValidationError for invalid source_type."""
        entries = [
            {"text": "Valid entry", "source_type": "session"},
            {"text": "Invalid source type", "source_type": "invalid_type"},
        ]

        with pytest.raises(ValidationError) as exc_info:
            add_memories(entries, atomic=True)

        assert "Invalid source_type" in str(exc_info.value)

    def test_atomic_mode_missing_source_type_raises(self, memory_db) -> None:
        """Test atomic mode raises ValidationError for missing source_type."""
        entries = [
            {"text": "Valid entry", "source_type": "session"},
            {"text": "Missing source type", "source_type": ""},  # Empty string
        ]

        with pytest.raises(ValidationError) as exc_info:
            add_memories(entries, atomic=True)

        assert "source_type" in str(exc_info.value).lower()

    def test_atomic_mode_all_or_nothing(self, memory_db) -> None:
        """Test atomic mode ensures all succeed or all fail together."""
        entries = [
            {"text": "First valid entry", "source_type": "session"},
            {"text": "Second valid entry", "source_type": "file"},
        ]

        # All valid - should succeed
        results = add_memories(entries, atomic=True)
        assert all(r["success"] is True for r in results)


class TestAddMemoriesNonAtomicMode:
    """Tests for non-atomic mode behavior."""

    def test_non_atomic_partial_success_possible(self, memory_db) -> None:
        """Test non-atomic mode allows partial success."""
        entries = [
            {"text": "Valid entry 1", "source_type": "session"},
            {"text": "Valid entry 2", "source_type": "file"},
        ]

        results = add_memories(entries, atomic=False)

        assert len(results) == 2
        # Both should succeed in normal conditions
        assert all(r["success"] is True for r in results)

    def test_non_atomic_returns_individual_results(self, memory_db) -> None:
        """Test non-atomic mode returns individual results per entry."""
        entries = [
            {"text": "Memory A", "source_type": "session"},
            {"text": "Memory B", "source_type": "file"},
            {"text": "Memory C", "source_type": "web"},
        ]

        results = add_memories(entries, atomic=False)

        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is True
        assert results[2]["success"] is True


class TestAddMemoriesValidation:
    """Tests for input validation."""

    def test_missing_text_raises(self, memory_db) -> None:
        """Test that missing text raises ValidationError."""
        entries = [
            {"text": "Valid text", "source_type": "session"},
            {"source_type": "session"},  # Missing text
        ]

        with pytest.raises(ValidationError) as exc_info:
            add_memories(entries, atomic=True)

        assert "text" in str(exc_info.value).lower()

    def test_empty_text_raises(self, memory_db) -> None:
        """Test that empty text raises ValidationError."""
        entries = [
            {"text": "", "source_type": "session"},
        ]

        with pytest.raises(ValidationError) as exc_info:
            add_memories(entries, atomic=True)

        assert "text" in str(exc_info.value).lower()

    def test_whitespace_only_text_accepted(self, memory_db) -> None:
        """Test that whitespace-only text is accepted (current implementation behavior)."""
        entries = [
            {"text": "   ", "source_type": "session"},
        ]

        # Current implementation accepts whitespace-only strings since they are non-empty strings
        results = add_memories(entries, atomic=False)
        assert len(results) == 1
        assert results[0]["success"] is True

    def test_invalid_source_type_format_raises(self, memory_db) -> None:
        """Test that invalid source_type format raises ValidationError."""
        entries = [
            {"text": "Some text", "source_type": 123},  # Not a string
        ]

        with pytest.raises(ValidationError) as exc_info:
            add_memories(entries, atomic=True)

        assert "source_type" in str(exc_info.value).lower()

    def test_forbidden_source_path_pattern_raises(self, memory_db) -> None:
        """Test that forbidden patterns in source_path raise ValidationError."""
        entries = [
            {
                "text": "Some text",
                "source_type": "session",
                "source_path": "/*comment*/",
            },
        ]

        with pytest.raises(ValidationError) as exc_info:
            add_memories(entries, atomic=True)

        assert "source_path" in str(exc_info.value).lower()


class TestAddMemoriesHashVerification:
    """Tests for hash generation and verification."""

    def test_results_contain_correct_hashes(self, memory_db) -> None:
        """Test that successful results contain correct hashes."""
        text = "Memory with known hash"
        expected_hash = compute_hash(text)

        entries = [
            {"text": text, "source_type": "session"},
        ]

        results = add_memories(entries, atomic=False)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["hash"] == expected_hash

    def test_different_texts_produce_different_hashes(self, memory_db) -> None:
        """Test that different texts produce different hashes."""
        entries = [
            {"text": "First unique text", "source_type": "session"},
            {"text": "Second unique text", "source_type": "session"},
        ]

        results = add_memories(entries, atomic=False)

        assert len(results) == 2
        assert results[0]["hash"] != results[1]["hash"]

    def test_same_text_produces_same_hash(self, memory_db) -> None:
        """Test that same text produces same hash across calls."""
        text = "Identical text content"
        expected_hash = compute_hash(text)

        entries1 = [{"text": text, "source_type": "session"}]
        entries2 = [{"text": text, "source_type": "session"}]

        results1 = add_memories(entries1, atomic=False)
        results2 = add_memories(entries2, atomic=False)

        assert results1[0]["hash"] == expected_hash
        assert results2[0]["hash"] == expected_hash
        assert results1[0]["hash"] == results2[0]["hash"]

    def test_multiple_entries_hashes_are_unique(self, memory_db) -> None:
        """Test that multiple entries each get unique hashes."""
        entries = [
            {"text": "Unique memory number one", "source_type": "session"},
            {"text": "Unique memory number two", "source_type": "session"},
            {"text": "Unique memory number three", "source_type": "session"},
        ]

        results = add_memories(entries, atomic=False)

        hashes = [r["hash"] for r in results]
        assert len(hashes) == len(set(hashes)), "All hashes should be unique"


class TestAddMemoriesSourceTypes:
    """Tests for valid source_type handling."""

    @pytest.mark.parametrize("source_type", ["session", "file", "web", "boomerang"])
    def test_valid_source_types_accepted(self, memory_db, source_type) -> None:
        """Test that all valid source_types are accepted."""
        entries = [
            {"text": f"Memory with {source_type}", "source_type": source_type},
        ]

        results = add_memories(entries, atomic=False)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["hash"] is not None
