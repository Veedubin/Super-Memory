"""Database migration module for Super-Memory v0.5.0.

This module provides migration functionality to move from the old single-table
format (v0.4.x) to the new dual-table format (v0.5.0).

Usage:
    # As a module within super_memory package:
    from super_memory.migrate import migrate_database

    migrate_database(
        old_db_path="./memory_data",
        new_db_path="./memory_data_v2",
        skip_bge=False,
        dry_run=False,
    )

    # Via CLI:
    python -m super_memory.migrate --old-db ./memory_data --new-db ./memory_data_v2
"""

import argparse
import datetime
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import lancedb

from .config import get_config
from .schema import _get_memory_schema, _get_memory_schema_long

logger = logging.getLogger(__name__)

MINILM_DIMS = 384
BGE_DIMS = 1024


class MigrationError(Exception):
    """Raised when migration fails."""
    pass


class MigrationStats:
    """Statistics collected during migration."""

    def __init__(self):
        self.total: int = 0
        self.minilm_success: int = 0
        self.minilm_failed: int = 0
        self.bge_success: int = 0
        self.bge_failed: int = 0
        self.skipped_corrupted: int = 0
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None

    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


def detect_old_db_model(old_db_path: str) -> str:
    """Detect which embedding model the old database was using.

    Args:
        old_db_path: Path to old database.

    Returns:
        "minilm" (384-dim), "bge" (1024-dim), or "unknown"
    """
    try:
        db = lancedb.connect(old_db_path)
        tables = db.list_tables()

        if "memories" not in tables:
            return "unknown"

        table = db.open_table("memories")
        schema = table.schema

        for field in schema:
            if field.name == "vector":
                dims = field.type.list_size
                if dims == MINILM_DIMS:
                    return "minilm"
                elif dims == BGE_DIMS:
                    return "bge"
                else:
                    logger.warning("Unknown vector dimensions: %d", dims)
                    return "unknown"

        return "unknown"
    except Exception as e:
        logger.error("Failed to detect old DB model: %s", e)
        return "unknown"


def create_backup(old_db_path: str) -> Optional[str]:
    """Create a timestamped backup of the old database.

    Args:
        old_db_path: Path to old database.

    Returns:
        Path to backup directory, or None if backup failed.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{old_db_path}_backup_{timestamp}"

    try:
        shutil.copytree(old_db_path, backup_path)
        logger.info("Backup created: %s", backup_path)
        return backup_path
    except Exception as e:
        logger.error("Failed to create backup: %s", e)
        return None


def load_old_memories(old_db_path: str) -> list[dict[str, Any]]:
    """Load all memories from the old database.

    Args:
        old_db_path: Path to old database.

    Returns:
        List of memory entries.

    Raises:
        MigrationError: If database cannot be read.
    """
    db = lancedb.connect(old_db_path)

    try:
        tables_resp = db.list_tables()
        # list_tables() returns ListTablesResponse object with .tables attribute
        tables = tables_resp.tables if hasattr(tables_resp, 'tables') else tables_resp
    except Exception as e:
        raise MigrationError(f"Cannot list tables (database may be corrupted): {e}")

    if "memories" not in tables:
        raise MigrationError("No 'memories' table found in old database")

    try:
        table = db.open_table("memories")
    except Exception as e:
        raise MigrationError(f"Cannot open 'memories' table (database may be corrupted): {e}")

    try:
        # Use to_pandas() to get all entries
        df = table.to_pandas()
        memories = df.to_dict("records")
        logger.info("Loaded %d memories from old database", len(memories))
        return memories
    except Exception as e:
        # Fallback: try to iterate with take_offsets() for rows we can read
        logger.warning("to_pandas() failed, trying take_offsets() approach: %s", e)
        try:
            # Get row count first
            total_rows = table.count_rows()
            # Try to read in batches using take_offsets
            all_memories = []
            batch_size = 1000
            for offset in range(0, total_rows, batch_size):
                batch_indices = list(range(offset, min(offset + batch_size, total_rows)))
                try:
                    batch_data = table.take_offsets(batch_indices).to_list()
                    all_memories.extend(batch_data)
                except Exception as batch_e:
                    logger.warning("Failed to read batch at offset %d: %s", offset, batch_e)
            logger.info("Loaded %d memories using take_offsets()", len(all_memories))
            return all_memories
        except Exception as e2:
            raise MigrationError(f"Cannot read table data: {e2}")


def migrate_memories(
    memories: list[dict[str, Any]],
    new_db_path: str,
    skip_bge: bool = False,
    dry_run: bool = False,
    stats: Optional[MigrationStats] = None,
) -> None:
    """Migrate memories to new dual-table format.

    Args:
        memories: List of memory entries from old DB.
        new_db_path: Path for new database.
        skip_bge: If True, skip BGE re-embedding.
        dry_run: If True, don't write anything.
        stats: Optional MigrationStats object to collect statistics.

    Raises:
        MigrationError: If migration fails.
    """
    if stats is None:
        stats = MigrationStats()

    os.makedirs(new_db_path, exist_ok=True)
    new_db = lancedb.connect(new_db_path)

    # Create tables with proper schemas
    if not dry_run:
        new_db.create_table("memories", schema=_get_memory_schema(), exist_ok=True)
        logger.info("Created/verified 'memories' table (MiniLM)")

        if not skip_bge:
            new_db.create_table("memories_long", schema=_get_memory_schema_long(), exist_ok=True)
            logger.info("Created/verified 'memories_long' table (BGE-Large)")

    new_table = new_db.open_table("memories")
    new_table_long = new_db.open_table("memories_long") if not skip_bge else None

    # Import sentence-transformers here to ensure it's available
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise MigrationError(
            "sentence-transformers is required for migration. "
            "Install with: pip install sentence-transformers"
        )

    # Initialize embedding models
    config = get_config()

    logger.info("Loading MiniLM model (%s)...", config.model)
    minilm_model = SentenceTransformer(config.model)

    bge_model = None
    if not skip_bge:
        logger.info("Loading BGE model (%s)...", config.bge_model)
        bge_model = SentenceTransformer(config.bge_model)

    total = len(memories)
    bge_eta_shown = False

    for i, memory in enumerate(memories):
        try:
            text = memory.get("text", "")
            if not text:
                stats.skipped_corrupted += 1
                continue

            # Prepare entry (normalize fields)
            entry = {
                "text": text,
                "source_type": memory.get("source_type", "session"),
                "source_path": memory.get("source_path"),
                "timestamp": memory.get("timestamp", datetime.datetime.now()),
                "content_hash": memory.get("content_hash"),
                "metadata_json": memory.get("metadata_json"),
            }

            # MiniLM embedding (fast)
            try:
                minilm_vector = minilm_model.encode([text])[0].tolist()
                entry_minilm = {**entry, "vector": minilm_vector}

                if not dry_run:
                    new_table.add([entry_minilm])

                stats.minilm_success += 1
            except Exception as e:
                logger.warning("Failed to embed with MiniLM: %s", e)
                stats.minilm_failed += 1

            # BGE embedding (slow)
            if not skip_bge and bge_model:
                try:
                    if not bge_eta_shown and total > 10:
                        estimated_time = total * 1.5  # ~1.5 sec per item
                        logger.info("BGE re-embedding: ~%.0f seconds estimated", estimated_time)
                        bge_eta_shown = True

                    bge_vector = bge_model.encode([text])[0].tolist()
                    entry_bge = {**entry, "vector": bge_vector}

                    if not dry_run:
                        new_table_long.add([entry_bge])

                    stats.bge_success += 1
                except Exception as e:
                    logger.warning("Failed to embed with BGE: %s", e)
                    stats.bge_failed += 1

            stats.total += 1

            # Progress logging
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(
                    "Progress: %d/%d (MiniLM: %d, BGE: %d)",
                    i + 1, total, stats.minilm_success, stats.bge_success
                )

        except Exception as e:
            logger.warning("Failed to process memory at index %d: %s", i, e)
            stats.skipped_corrupted += 1

    logger.info(
        "Migration complete: %d processed, %d MiniLM success, %d BGE success",
        stats.total, stats.minilm_success, stats.bge_success
    )


def verify_migration(old_db_path: str, new_db_path: str, skip_bge: bool) -> dict[str, int]:
    """Verify migration by comparing row counts.

    Args:
        old_db_path: Path to old database.
        new_db_path: Path to new database.
        skip_bge: Whether BGE was skipped.

    Returns:
        Dict with counts: old_count, new_minilm_count, new_bge_count.
    """
    result = {}

    try:
        old_db = lancedb.connect(old_db_path)
        old_table = old_db.open_table("memories")
        result["old_count"] = old_table.count_rows()
    except Exception as e:
        logger.warning("Cannot verify old DB count: %s", e)
        result["old_count"] = -1

    try:
        new_db = lancedb.connect(new_db_path)
        new_table = new_db.open_table("memories")
        result["new_minilm_count"] = new_table.count_rows()
    except Exception as e:
        logger.warning("Cannot verify new MiniLM count: %s", e)
        result["new_minilm_count"] = -1

    if not skip_bge:
        try:
            new_db = lancedb.connect(new_db_path)
            new_table_long = new_db.open_table("memories_long")
            result["new_bge_count"] = new_table_long.count_rows()
        except Exception as e:
            logger.warning("Cannot verify new BGE count: %s", e)
            result["new_bge_count"] = -1
    else:
        result["new_bge_count"] = 0

    return result


def print_summary(stats: MigrationStats, verify_result: dict[str, int], dry_run: bool, skip_bge: bool) -> None:
    """Print migration summary report.

    Args:
        stats: Migration statistics.
        verify_result: Verification counts.
        dry_run: Whether this was a dry run.
        skip_bge: Whether BGE was skipped.
    """
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)

    if dry_run:
        print("[DRY RUN - No data was written]")

    print(f"\nProcessed: {stats.total} memories")
    print(f"  MiniLM success: {stats.minilm_success}")
    print(f"  MiniLM failed:  {stats.minilm_failed}")
    if not skip_bge:
        print(f"  BGE success:    {stats.bge_success}")
        print(f"  BGE failed:     {stats.bge_failed}")
    print(f"  Skipped (corrupted): {stats.skipped_corrupted}")

    print(f"\nDuration: {stats.duration_seconds():.1f} seconds")

    print("\nVerification:")
    print(f"  Old DB count:        {verify_result.get('old_count', 'N/A')}")
    print(f"  New MiniLM count:   {verify_result.get('new_minilm_count', 'N/A')}")
    if not skip_bge:
        print(f"  New BGE count:      {verify_result.get('new_bge_count', 'N/A')}")

    # Check for issues
    issues = []
    if stats.minilm_failed > 0:
        issues.append(f"{stats.minilm_failed} memories failed MiniLM embedding")
    if stats.bge_failed > 0 and not skip_bge:
        issues.append(f"{stats.bge_failed} memories failed BGE embedding")
    if stats.skipped_corrupted > 0:
        issues.append(f"{stats.skipped_corrupted} corrupted memories skipped")

    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Migration completed successfully!")

    print("=" * 60)


def migrate_database(
    old_db_path: str,
    new_db_path: str,
    skip_bge: bool = False,
    dry_run: bool = False,
    backup: bool = True,
    verbose: bool = False,
) -> MigrationStats:
    """Migrate database from old single-table to new dual-table format.

    Args:
        old_db_path: Path to old database directory.
        new_db_path: Path for new database directory.
        skip_bge: If True, skip BGE re-embedding (faster).
        dry_run: If True, don't write anything.
        backup: If True, create backup of old database.
        verbose: If True, enable verbose logging.

    Returns:
        MigrationStats object with migration statistics.

    Raises:
        MigrationError: If migration fails.

    Example:
        >>> from super_memory.migrate import migrate_database
        >>> stats = migrate_database(
        ...     old_db_path="./memory_data",
        ...     new_db_path="./memory_data_v2",
        ...     skip_bge=False,
        ... )
        >>> print(f"Migrated {stats.total} memories")
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    old_db_path = os.path.abspath(old_db_path)
    new_db_path = os.path.abspath(new_db_path)

    if not os.path.exists(old_db_path):
        raise MigrationError(f"Old database path does not exist: {old_db_path}")

    if os.path.exists(new_db_path) and os.listdir(new_db_path):
        raise MigrationError(
            f"New database path exists and is not empty: {new_db_path}. "
            "Please use a different path or remove the existing directory."
        )

    logger.info("Starting migration: %s -> %s", old_db_path, new_db_path)

    # Detect old database model
    detected_model = detect_old_db_model(old_db_path)
    logger.info("Detected old DB model: %s", detected_model)

    if detected_model == "unknown":
        logger.warning("Could not detect old model. Proceeding anyway...")

    # Create backup
    if not dry_run and backup:
        logger.info("Creating backup...")
        backup_path = create_backup(old_db_path)
        if backup_path is None:
            logger.warning("Backup failed. Continuing anyway...")

    # Load old memories
    logger.info("Loading memories from old database...")
    try:
        memories = load_old_memories(old_db_path)
    except MigrationError:
        raise
    except Exception as e:
        raise MigrationError(f"Failed to load old database: {e}") from e

    if not memories:
        logger.info("No memories found in old database.")
        stats = MigrationStats()
        stats.end_time = datetime.datetime.now()
        return stats

    logger.info("Found %d memories to migrate", len(memories))

    # Migrate
    stats = MigrationStats()
    stats.start_time = datetime.datetime.now()

    try:
        migrate_memories(
            memories=memories,
            new_db_path=new_db_path,
            skip_bge=skip_bge,
            dry_run=dry_run,
            stats=stats,
        )
    except MigrationError:
        raise
    except Exception as e:
        raise MigrationError(f"Migration failed: {e}") from e

    stats.end_time = datetime.datetime.now()

    # Verify
    logger.info("Verifying migration...")
    try:
        verify_result = verify_migration(old_db_path, new_db_path, skip_bge)
    except Exception as e:
        logger.warning("Verification failed: %s", e)
        verify_result = {}

    # Print summary
    print_summary(stats, verify_result, dry_run, skip_bge)

    if not dry_run:
        logger.info("New database location: %s", new_db_path)
        logger.info(
            "To use the new database, set SUPER_MEMORY_DB_PATH environment variable:\n"
            "  export SUPER_MEMORY_DB_PATH=%s",
            new_db_path
        )

    return stats


def main() -> None:
    """CLI entry point for migration."""
    from .config import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Super-Memory Database Migration Script v0.5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Migrate to new database:
    python -m super_memory.migrate --old-db ./memory_data --new-db ./memory_data_v2

  Skip BGE re-embedding (faster):
    python -m super_memory.migrate --old-db ./memory_data --new-db ./memory_data_v2 --skip-bge

  Preview migration without writing:
    python -m super_memory.migrate --old-db ./memory_data --dry-run

  Disable automatic backup:
    python -m super_memory.migrate --old-db ./memory_data --new-db ./memory_data_v2 --no-backup
        """,
    )

    parser.add_argument(
        "--old-db",
        required=True,
        help="Path to old database directory",
    )
    parser.add_argument(
        "--new-db",
        required=True,
        help="Path for new database directory",
    )
    parser.add_argument(
        "--skip-bge",
        action="store_true",
        help="Skip BGE re-embedding (faster, only creates MiniLM table)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without writing",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic backup of old database",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Super-Memory Database Migration Script v0.5.0")
    print("=" * 60)
    print(f"Old DB:  {os.path.abspath(args.old_db)}")
    print(f"New DB:  {os.path.abspath(args.new_db)}")
    print(f"Skip BGE: {args.skip_bge}")
    print(f"Dry run:  {args.dry_run}")
    print("=" * 60)

    try:
        migrate_database(
            old_db_path=args.old_db,
            new_db_path=args.new_db,
            skip_bge=args.skip_bge,
            dry_run=args.dry_run,
            backup=not args.no_backup,
            verbose=args.verbose,
        )
    except MigrationError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nMigration cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
