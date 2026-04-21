"""Exception hierarchy for Super-Memory.

All exceptions inherit from SuperMemoryError so callers can catch everything
with a single except clause, or be specific as needed.
"""

from typing import Any, Optional


class SuperMemoryError(Exception):
    """Base exception for all Super-Memory errors.

    Attributes:
        message: Human-readable error description.
        details: Optional structured data for debugging (e.g., query, path).
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class DatabaseError(SuperMemoryError):
    """Raised when LanceDB connection or table operations fail.

    Examples: cannot connect, cannot open table, cannot add rows.
    """


class TableNotFoundError(DatabaseError):
    """Raised when a table that should exist cannot be found."""


class MigrationError(DatabaseError):
    """Raised when schema migration fails."""


class QueryError(SuperMemoryError):
    """Raised when a search or filter query fails."""


class MemoryNotFoundError(SuperMemoryError):
    """Raised when an operation expects a memory to exist but it does not.

    Note: Read operations like recall_memory_by_path should return None,
    not raise this. This is for write operations (e.g., delete, update).
    """


class ValidationError(SuperMemoryError):
    """Raised when input validation fails before touching the database."""


class ConfigurationError(SuperMemoryError):
    """Raised when configuration is invalid or missing."""
