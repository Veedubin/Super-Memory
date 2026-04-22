# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-04-22

### Added

- Table corruption detection via `_validate_table_accessible()` in `memory.py`
- New MCP search tool: `boomerang_memory_search_tiered`
- New MCP search tool: `boomerang_memory_search_parallel`

### Removed

- Migration script `migrate.py` (no longer needed post-v0.5.0)
- Migration script `migrate_to_minilm.py` (no longer needed post-v0.5.0)
- Migration script `scripts/migrate_db.py` (no longer needed post-v0.5.0)

### Changed

- Style and formatting fixes
