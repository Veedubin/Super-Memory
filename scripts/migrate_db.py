#!/usr/bin/env python3
"""
Super-Memory Database Migration Script v0.5.0

Migrates old single-table databases to new dual-table format.
This is a standalone script that can be copied to any project and run independently.

Usage:
    python migrate_db.py --old-db ./memory_data --new-db ./memory_data_v2
    python migrate_db.py --old-db ./memory_data --skip-bge  # Skip BGE re-embedding
    python migrate_db.py --old-db ./memory_data --dry-run   # Preview only

Requirements:
    pip install lancedb lance sentence-transformers tqdm
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import lancedb
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector
except ImportError:
    print("ERROR: lancedb is required. Install with: pip install lancedb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is required. Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("migrate_db")


# Configuration
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
MINILM_DIMS = 384
BGE_DIMS = 1024


@dataclass
class MigrationStats:
    """Statistics collected during migration."""
    total: int = 0
    minilm_success: int = 0
    minilm_failed: int = 0
    bge_success: int = 0
    bge_failed: int = 0
    skipped_corrupted: int = 0
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None

    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class StandaloneSchema:
    """Standalone schema definitions without relying on super_memory package."""

    @staticmethod
    def create_minilm_schema():
        """Create MiniLM schema (384-dim)."""
        registry = get_registry().get("sentence-transformers")
        embed_model = registry.create(name=MINILM_MODEL)

        class MemorySchema(LanceModel):
            text: str = embed_model.SourceField()
            vector: Vector(embed_model.ndims()) = embed_model.VectorField()  # type: ignore
            source_type: str = "session"
            source_path: Optional[str] = None
            timestamp: datetime.datetime
            content_hash: Optional[str] = None
            metadata_json: Optional[str] = None

        return MemorySchema, embed_model

    @staticmethod
    def create_bge_schema():
        """Create BGE-Large schema (1024-dim)."""
        registry = get_registry().get("sentence-transformers")
        embed_model = registry.create(name=BGE_MODEL)

        class MemorySchemaLong(LanceModel):
            text: str = embed_model.SourceField()
            vector: Vector(embed_model.ndims()) = embed_model.VectorField()  # type: ignore
            source_type: str = "session"
            source_path: Optional[str] = None
            timestamp: datetime.datetime
            content_hash: Optional[str] = None
            metadata_json: Optional[str] = None

        return MemorySchemaLong, embed_model


def detect_old_db_model(old_db_path: str) -> str:
    """Detect which embedding model the old database was using.

    Args:
        old_db_path: Path to old database.

    Returns:
        "minilm" (384-dim), "bge" (1024-dim), or "unknown"
    """
    try:
        db = lancedb.connect(old_db_path)
        tables_resp = db.list_tables()

        # list_tables() returns ListTablesResponse object with .tables attribute
        tables = tables_resp.tables if hasattr(tables_resp, 'tables') else tables_resp

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
                    logger.warning(f"Unknown vector dimensions: {dims}")
                    return "unknown"

        return "unknown"
    except Exception as e:
        logger.error(f"Failed to detect old DB model: {e}")
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
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None


def load_old_memories(old_db_path: str) -> list[dict[str, Any]]:
    """Load all memories from the old database.

    Args:
        old_db_path: Path to old database.

    Returns:
        List of memory entries.

    Raises:
        Exception: If database cannot be read.
    """
    db = lancedb.connect(old_db_path)

    try:
        tables_resp = db.list_tables()
        # list_tables() returns ListTablesResponse object with .tables attribute
        tables = tables_resp.tables if hasattr(tables_resp, 'tables') else tables_resp
    except Exception as e:
        raise Exception(f"Cannot list tables (database may be corrupted): {e}")

    if "memories" not in tables:
        raise Exception("No 'memories' table found in old database")

    try:
        table = db.open_table("memories")
    except Exception as e:
        raise Exception(f"Cannot open 'memories' table (database may be corrupted): {e}")

    try:
        # Use to_pandas() to get all entries, handling corrupted fragments gracefully
        df = table.to_pandas()
        memories = df.to_dict("records")
        logger.info(f"Loaded {len(memories)} memories from old database")
        return memories
    except Exception as e:
        # Fallback: try to iterate with take_offsets() for rows we can read
        logger.warning(f"to_pandas() failed, trying take_offsets() approach: {e}")
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
                    logger.warning(f"Failed to read batch at offset {offset}: {batch_e}")
            logger.info(f"Loaded {len(all_memories)} memories using take_offsets()")
            return all_memories
        except Exception as e2:
            raise Exception(f"Cannot read table data: {e2}")


def migrate_memories(
    memories: list[dict[str, Any]],
    new_db_path: str,
    skip_bge: bool,
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """Migrate memories to new dual-table format.

    Args:
        memories: List of memory entries from old DB.
        new_db_path: Path for new database.
        skip_bge: If True, skip BGE re-embedding.
        dry_run: If True, don't write anything.
        stats: Migration statistics object.
    """
    os.makedirs(new_db_path, exist_ok=True)
    new_db = lancedb.connect(new_db_path)

    # Initialize embedding models
    logger.info("Loading MiniLM model...")
    minilm_model = SentenceTransformer(MINILM_MODEL)
    minilm_dims = MINILM_DIMS

    if not skip_bge:
        logger.info("Loading BGE model (this may take a minute)...")
        bge_model = SentenceTransformer(BGE_MODEL)
        bge_dims = BGE_DIMS
    else:
        bge_model = None
        bge_dims = 0

    # Create tables
    if not dry_run:
        registry = get_registry().get("sentence-transformers")
        minilm_embed = registry.create(name=MINILM_MODEL)

        class MiniLMSchema(LanceModel):
            text: str = minilm_embed.SourceField()
            vector: Vector(minilm_embed.ndims()) = minilm_embed.VectorField()  # type: ignore
            source_type: str = "session"
            source_path: Optional[str] = None
            timestamp: datetime.datetime
            content_hash: Optional[str] = None
            metadata_json: Optional[str] = None

        new_db.create_table("memories", schema=MiniLMSchema, exist_ok=True)

        if not skip_bge:
            bge_embed = registry.create(name=BGE_MODEL)

            class BGESchema(LanceModel):
                text: str = bge_embed.SourceField()
                vector: Vector(bge_embed.ndims()) = bge_embed.VectorField()  # type: ignore
                source_type: str = "session"
                source_path: Optional[str] = None
                timestamp: datetime.datetime
                content_hash: Optional[str] = None
                metadata_json: Optional[str] = None

            new_db.create_table("memories_long", schema=BGESchema, exist_ok=True)

    new_table = new_db.open_table("memories")
    new_table_long = new_db.open_table("memories_long") if not skip_bge else None

    # Progress bar
    if tqdm:
        pbar = tqdm(total=len(memories), desc="Migrating", unit="mem")
    else:
        pbar = None

    bge_etaShown = False

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
                logger.warning(f"Failed to embed with MiniLM: {e}")
                stats.minilm_failed += 1

            # BGE embedding (slow)
            if not skip_bge and bge_model:
                try:
                    if not bge_etaShown and len(memories) > 10:
                        estimated_time = len(memories) * 1.5  # ~1.5 sec per item
                        logger.info(f"BGE re-embedding: ~{estimated_time:.0f} seconds estimated")
                        bge_etaShown = True

                    bge_vector = bge_model.encode([text])[0].tolist()
                    entry_bge = {**entry, "vector": bge_vector}

                    if not dry_run:
                        new_table_long.add([entry_bge])

                    stats.bge_success += 1
                except Exception as e:
                    logger.warning(f"Failed to embed with BGE: {e}")
                    stats.bge_failed += 1

            stats.total += 1

        except Exception as e:
            logger.warning(f"Failed to process memory at index {i}: {e}")
            stats.skipped_corrupted += 1

        if pbar:
            update_dict = {"MiniLM": f"{stats.minilm_success}/{stats.total}"}
            if not skip_bge:
                update_dict["BGE"] = f"{stats.bge_success}/{stats.total}"
            pbar.update(1)
            pbar.set_postfix(**update_dict)

    if pbar:
        pbar.close()


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
        logger.warning(f"Cannot verify old DB count: {e}")
        result["old_count"] = -1

    try:
        new_db = lancedb.connect(new_db_path)
        new_table = new_db.open_table("memories")
        result["new_minilm_count"] = new_table.count_rows()
    except Exception as e:
        logger.warning(f"Cannot verify new MiniLM count: {e}")
        result["new_minilm_count"] = -1

    if not skip_bge:
        try:
            new_db = lancedb.connect(new_db_path)
            new_table_long = new_db.open_table("memories_long")
            result["new_bge_count"] = new_table_long.count_rows()
        except Exception as e:
            logger.warning(f"Cannot verify new BGE count: {e}")
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
    print(f"  New MiniLM count:    {verify_result.get('new_minilm_count', 'N/A')}")
    if not skip_bge:
        print(f"  New BGE count:       {verify_result.get('new_bge_count', 'N/A')}")

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


def main():
    """Main entry point for standalone migration script."""
    parser = argparse.ArgumentParser(
        description="Super-Memory Database Migration Script v0.5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Migrate to new database:
    python migrate_db.py --old-db ./memory_data --new-db ./memory_data_v2

  Skip BGE re-embedding (faster):
    python migrate_db.py --old-db ./memory_data --new-db ./memory_data_v2 --skip-bge

  Preview migration without writing:
    python migrate_db.py --old-db ./memory_data --dry-run

  Disable automatic backup:
    python migrate_db.py --old-db ./memory_data --new-db ./memory_data_v2 --no-backup
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

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    old_db_path = os.path.abspath(args.old_db)
    new_db_path = os.path.abspath(args.new_db)

    if not os.path.exists(old_db_path):
        print(f"ERROR: Old database path does not exist: {old_db_path}")
        sys.exit(1)

    if os.path.exists(new_db_path) and os.listdir(new_db_path):
        print(f"ERROR: New database path exists and is not empty: {new_db_path}")
        print("Please use a different path or remove the existing directory.")
        sys.exit(1)

    print("=" * 60)
    print("Super-Memory Database Migration Script v0.5.0")
    print("=" * 60)
    print(f"Old DB:  {old_db_path}")
    print(f"New DB:  {new_db_path}")
    print(f"Skip BGE: {args.skip_bge}")
    print(f"Dry run:  {args.dry_run}")
    print("=" * 60)

    # Detect old database model
    print("\nDetecting old database model...")
    detected_model = detect_old_db_model(old_db_path)
    print(f"Detected model: {detected_model}")

    if detected_model == "unknown":
        print("WARNING: Could not detect old model. Proceeding anyway...")
    elif detected_model == "minilm":
        print("Old database used MiniLM (384-dim)")
    elif detected_model == "bge":
        print("Old database used BGE-Large (1024-dim)")

    # Create backup
    if not args.dry_run and not args.no_backup:
        print("\nCreating backup...")
        backup_path = create_backup(old_db_path)
        if backup_path is None:
            print("WARNING: Backup failed. Continuing anyway...")
        else:
            print(f"Backup saved to: {backup_path}")

    # Load old memories
    print("\nLoading memories from old database...")
    try:
        memories = load_old_memories(old_db_path)
    except Exception as e:
        print(f"ERROR: Failed to load old database: {e}")
        sys.exit(1)

    if not memories:
        print("No memories found in old database.")
        sys.exit(0)

    print(f"Found {len(memories)} memories to migrate")

    # Migrate
    stats = MigrationStats()
    stats.start_time = datetime.datetime.now()

    print(f"\nStarting migration (Skip BGE: {args.skip_bge})...")
    try:
        migrate_memories(
            memories=memories,
            new_db_path=new_db_path,
            skip_bge=args.skip_bge,
            dry_run=args.dry_run,
            stats=stats,
        )
    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        if not args.dry_run:
            print(f"\nRecovery suggestion: If migration failed partially, check {new_db_path}")
            print("The old database is still intact in the backup.")
        sys.exit(1)

    stats.end_time = datetime.datetime.now()

    # Verify
    print("\nVerifying migration...")
    try:
        verify_result = verify_migration(old_db_path, new_db_path, args.skip_bge)
    except Exception as e:
        logger.warning(f"Verification failed: {e}")
        verify_result = {}

    # Print summary
    print_summary(stats, verify_result, args.dry_run, args.skip_bge)

    if not args.dry_run:
        print(f"\nNew database location: {new_db_path}")
        print("\nTo use the new database, set SUPER_MEMORY_DB_PATH environment variable:")
        print(f"  export SUPER_MEMORY_DB_PATH={new_db_path}")


if __name__ == "__main__":
    main()
