"""Tests for the Super-Memory exception hierarchy."""

import pytest

from super_memory.exceptions import (
    SuperMemoryError,
    DatabaseError,
    TableNotFoundError,
    MigrationError,
    QueryError,
    MemoryNotFoundError,
    ValidationError,
    ConfigurationError,
)


class TestExceptionInstantiation:
    """Test that all exceptions can be instantiated with just a message."""

    @pytest.mark.parametrize(
        "exception_cls",
        [
            SuperMemoryError,
            DatabaseError,
            TableNotFoundError,
            MigrationError,
            QueryError,
            MemoryNotFoundError,
            ValidationError,
            ConfigurationError,
        ],
    )
    def test_instantiate_with_message_only(self, exception_cls):
        exc = exception_cls("Something went wrong")
        assert exc.message == "Something went wrong"
        assert exc.details == {}

    @pytest.mark.parametrize(
        "exception_cls",
        [
            SuperMemoryError,
            DatabaseError,
            TableNotFoundError,
            MigrationError,
            QueryError,
            MemoryNotFoundError,
            ValidationError,
            ConfigurationError,
        ],
    )
    def test_instantiate_with_details(self, exception_cls):
        details = {"key": "value", "count": 42}
        exc = exception_cls("Error with details", details=details)
        assert exc.message == "Error with details"
        assert exc.details == details


class TestInheritanceHierarchy:
    """Test that the exception inheritance hierarchy is correct."""

    def test_database_error_inherits_from_super_memory_error(self):
        assert issubclass(DatabaseError, SuperMemoryError)

    def test_table_not_found_error_inherits_from_database_error(self):
        assert issubclass(TableNotFoundError, DatabaseError)

    def test_migration_error_inherits_from_database_error(self):
        assert issubclass(MigrationError, DatabaseError)

    def test_query_error_inherits_from_super_memory_error(self):
        assert issubclass(QueryError, SuperMemoryError)

    def test_memory_not_found_error_inherits_from_super_memory_error(self):
        assert issubclass(MemoryNotFoundError, SuperMemoryError)

    def test_validation_error_inherits_from_super_memory_error(self):
        assert issubclass(ValidationError, SuperMemoryError)

    def test_configuration_error_inherits_from_super_memory_error(self):
        assert issubclass(ConfigurationError, SuperMemoryError)

    @pytest.mark.parametrize(
        "exception_cls",
        [
            DatabaseError,
            TableNotFoundError,
            MigrationError,
            QueryError,
            MemoryNotFoundError,
            ValidationError,
            ConfigurationError,
        ],
    )
    def test_all_exceptions_inherit_from_base(self, exception_cls):
        """Verify every exception in the hierarchy is a SuperMemoryError."""
        assert issubclass(exception_cls, SuperMemoryError)


class TestStrFormatting:
    """Test __str__ formatting works correctly."""

    def test_str_without_details(self):
        exc = SuperMemoryError("Simple error")
        assert str(exc) == "Simple error"

    def test_str_with_details(self):
        exc = SuperMemoryError("Error with details", details={"foo": "bar"})
        assert str(exc) == "Error with details (details: {'foo': 'bar'})"

    def test_str_with_empty_details(self):
        """Empty details should not trigger the details branch."""
        exc = SuperMemoryError("Error", details={})
        assert str(exc) == "Error"

    @pytest.mark.parametrize(
        "exception_cls",
        [
            DatabaseError,
            TableNotFoundError,
            MigrationError,
            QueryError,
            MemoryNotFoundError,
            ValidationError,
            ConfigurationError,
        ],
    )
    def test_str_without_details_all_subclasses(self, exception_cls):
        exc = exception_cls("Test error")
        assert str(exc) == "Test error"

    @pytest.mark.parametrize(
        "exception_cls",
        [
            DatabaseError,
            TableNotFoundError,
            MigrationError,
            QueryError,
            MemoryNotFoundError,
            ValidationError,
            ConfigurationError,
        ],
    )
    def test_str_with_details_all_subclasses(self, exception_cls):
        exc = exception_cls("Test error", details={"key": "value"})
        assert str(exc) == "Test error (details: {'key': 'value'})"


class TestExceptionCatching:
    """Test that exceptions can be caught at different levels of the hierarchy."""

    def test_catch_database_error_as_super_memory_error(self):
        with pytest.raises(SuperMemoryError):
            raise DatabaseError("DB failed")

    def test_catch_table_not_found_as_database_error(self):
        with pytest.raises(DatabaseError):
            raise TableNotFoundError("Table missing")

    def test_catch_table_not_found_as_super_memory_error(self):
        """TableNotFoundError can be caught as SuperMemoryError."""
        with pytest.raises(SuperMemoryError):
            raise TableNotFoundError("Table missing")

    def test_catch_migration_error_as_database_error(self):
        with pytest.raises(DatabaseError):
            raise MigrationError("Migration failed")

    def test_catch_migration_error_as_super_memory_error(self):
        with pytest.raises(SuperMemoryError):
            raise MigrationError("Migration failed")

    def test_catch_query_error_as_super_memory_error(self):
        with pytest.raises(SuperMemoryError):
            raise QueryError("Query failed")

    def test_catch_memory_not_found_as_super_memory_error(self):
        with pytest.raises(SuperMemoryError):
            raise MemoryNotFoundError("Memory not found")

    def test_catch_validation_error_as_super_memory_error(self):
        with pytest.raises(SuperMemoryError):
            raise ValidationError("Validation failed")

    def test_catch_configuration_error_as_super_memory_error(self):
        with pytest.raises(SuperMemoryError):
            raise ConfigurationError("Config invalid")

    def test_catch_all_exceptions_as_base(self):
        """All exceptions can be caught with a single SuperMemoryError handler."""
        for exc_cls in [
            DatabaseError,
            TableNotFoundError,
            MigrationError,
            QueryError,
            MemoryNotFoundError,
            ValidationError,
            ConfigurationError,
        ]:
            with pytest.raises(SuperMemoryError):
                raise exc_cls("Test error")

    def test_isinstance_check(self):
        """Verify isinstance checks work correctly."""
        table_not_found = TableNotFoundError("Table not found")
        assert isinstance(table_not_found, TableNotFoundError)
        assert isinstance(table_not_found, DatabaseError)
        assert isinstance(table_not_found, SuperMemoryError)
        assert not isinstance(table_not_found, QueryError)
