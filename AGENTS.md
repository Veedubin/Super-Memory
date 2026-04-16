# Agent Instructions for Super-Memory

## Build/Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

### Setup
```bash
# Sync dependencies
uv sync

# Run the main application
uv run -m super_memory
```

### Testing
```bash
# Run all tests (when tests are added)
uv run pytest

# Run a single test file
uv run pytest path/to/test_file.py

# Run a single test function
uv run pytest path/to/test_file.py::test_function_name

# Run with coverage
uv run pytest --cov
```

### Linting & Formatting
```bash
# Format code with ruff
uv run ruff format .

# Check formatting (CI)
uv run ruff format --check .

# Lint with ruff
uv run ruff check .

# Fix auto-fixable linting issues
uv run ruff check --fix .

# Type checking with mypy (if added)
uv run mypy .
```

## Code Style Guidelines

### General
- **Python Version**: 3.13+
- **Formatter**: Ruff (compatible with Black)
- **Linter**: Ruff
- **Line Length**: 88 characters
- **Indent**: 4 spaces (spaces, not tabs)

### Imports
```python
# Group imports: stdlib first, third-party second, local third
import os
from typing import Optional

import requests

from myproject.module import helper
```
- Use absolute imports over relative imports
- Sort imports with `isort` compatible ordering (via ruff)
- Avoid wildcard imports (`from module import *`)

### Type Hints
- Use type hints for all function parameters and return types
- Use `Optional[T]` instead of `T | None` for Python < 3.10 compatibility
- Import types from `typing` module for complex types

```python
def process_data(input_data: list[str], threshold: int = 10) -> dict[str, int]:
    ...
```

### Naming Conventions
- **Modules**: `lowercase_with_underscores.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `lowercase_with_underscores`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Private**: `_leading_underscore` for internal use
- **Type Variables**: `PascalCase` or `T`, `K`, `V` for generics

### Error Handling
- Use specific exceptions, not bare `except:`
- Prefer `try/except/else/finally` blocks appropriately
- Use `raise from` when re-raising exceptions to preserve context

```python
try:
    result = risky_operation()
except ValueError as e:
    raise ProcessingError("Failed to process") from e
```

### Functions
- Keep functions focused and under 50 lines when possible
- Use docstrings for all public functions
- Follow Google-style or NumPy-style docstrings

```python
def calculate_total(items: list[Item]) -> float:
    """Calculate the total price of all items.
    
    Args:
        items: List of items to sum.
        
    Returns:
        Total price as a float.
        
    Raises:
        ValueError: If items list is empty.
    """
    if not items:
        raise ValueError("Items list cannot be empty")
    return sum(item.price for item in items)
```

### Comments
- Write self-documenting code; prefer clear naming over comments
- Use comments to explain "why", not "what"
- Keep comments up-to-date with code changes

### Testing
- Use pytest for all tests
- Name test files `test_*.py` or `*_test.py`
- Name test functions `test_*`
- Use descriptive test names that explain the behavior
- Use fixtures for common setup
- Aim for high test coverage on critical paths

### Project Structure
```
.
├── pyproject.toml       # Project configuration
├── .python-version      # Python version
├── README.md
├── src/                 # Source code
│   └── super_memory/
├── tests/               # Test files
└── AGENTS.md           # This file
```

### Before Committing
```bash
# Run all quality checks
uv run ruff format . && uv run ruff check . && uv run pytest
```

## Dependencies

Add new dependencies via:
```bash
uv add package-name
```

Add dev dependencies:
```bash
uv add --dev package-name
```
