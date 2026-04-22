#!/usr/bin/env python3
"""
Standalone migration script: BGE-large-en-v1.5 (1024 dims) → all-MiniLM-L6-v2 (384 dims)

Reads ALL entries from the old LanceDB and re-inserts them with new embeddings.
Does NOT import from super_memory package to avoid config/embedding function conflicts.

Usage:
    python migrate_to_minilm.py [--dry-run]

Note: The source database may have some corrupted fragments. The script will
attempt to read as much data as possible and report any issues.
"""

import argparse
import datetime
import os
import sys
import warnings
from pathlib import Path

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

OLD_DB_PATH = "./memory_data"
NEW_DB_PATH = "./memory_data_minilm"
NEW_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEW_DIMS = 384
BATCH_SIZE = 50


def get_embedding_function(device: str = "cuda"):
    """Create MiniLM embedding function with specified device."""
    registry = get_registry()
    model = registry.get("sentence-transformers").create(name=NEW_MODEL, device=device)
    return model


def detect_device():
    """Detect GPU availability, fallback to CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _get_tables_list(db):
    """Get list of tables from a database, handling response object types."""
    tables_response = db.list_tables()
    if hasattr(tables_response, "tables"):
        return tables_response.tables
    return list(tables_response)


def count_old_entries(db_path: str) -> tuple[int, list]:
    """Count entries and get version info from old database."""
    print(f"\n[DRY-RUN] Analyzing database: {db_path}")

    db = lancedb.connect(db_path)
    tables = _get_tables_list(db)
    if not tables:
        print("  No tables found.")
        return 0, []

    table_name = tables[0]
    print(f"  Opening table: {table_name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            table = db.open_table(table_name)
            count = table.count_rows()
            print(f"  Found {count} entries (via count_rows)")

            # Get version info
            versions = table.list_versions()
            print(f"  Database has {len(versions)} versions")

            return count, versions
        except Exception as e:
            print(f"  Error accessing table: {e}")
            return 0, []


def read_old_data(db_path: str) -> tuple:
    """Read all data from old database.

    Returns:
        tuple: (dataframe, error_list) where error_list contains fragment errors
    """
    print(f"\nReading old database: {db_path}")

    db = lancedb.connect(db_path)
    tables = _get_tables_list(db)
    if not tables:
        raise ValueError(f"No tables found in {db_path}")

    table_name = tables[0]
    print(f"  Opening table: {table_name}")

    errors = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = db.open_table(table_name)

    # Get schema for reference
    schema = table.schema
    print(f"  Schema: {len(schema)} fields")
    field_names = [f.name for f in schema]
    print(f"  Fields: {field_names}")

    # Get the row count first to compare with what we can actually read
    try:
        reported_count = table.count_rows()
        print(f"  Table reports {reported_count} total rows")
    except Exception as count_error:
        print(f"  Could not get row count: {count_error}")
        reported_count = 0

    # Try to read data
    df = None
    try:
        arrow_table = table.to_arrow()
        df = arrow_table.to_pandas()
        print(f"  Successfully read {len(df)} rows")
    except Exception as e:
        error_msg = str(e)
        print(f"  Error reading full table: {error_msg[:100]}...")

        # Extract which fragment is missing
        missing_path = None
        if "Not found:" in error_msg:
            missing_path = error_msg.split("Not found:")[1].split(",")[0].strip()
            print(f"  Missing fragment: {missing_path}")
            errors.append(missing_path)

        # The table reports entries but we can't read them
        if reported_count > 0:
            print(
                f"\n  WARNING: Table reports {reported_count} rows but cannot read all data."
            )
            print("  This indicates corrupted or missing data fragments.")
            print("\n  Database corruption detected:")
            print(f"    - Reported rows: {reported_count}")
            print(f"    - Readable rows: 0")
            print(
                f"    - Missing fragment: {missing_path.split('/')[-1] if missing_path else 'unknown'}"
            )

            # Try head() as a fallback
            print("\n  Attempting head() as fallback...")
            try:
                head_df = table.head(20).to_pandas()
                if len(head_df) > 0:
                    df = head_df
                    print(f"  Recovered {len(df)} rows via head() method")
                    print(
                        f"  WARNING: This is only a partial recovery. {reported_count - len(df)} entries are still missing."
                    )
                    return df, errors
            except Exception as head_error:
                print(f"  Head also failed: {head_error}")

            print("\n  ERROR: Cannot recover data from corrupted database.")
            print(
                "  This database has corrupted fragments and requires pylance for recovery."
            )
            print("  Install with: pip install pylance")
            print("\n  Alternative solutions:")
            print("    1. Restore from a backup")
            print("    2. Recreate the database from source")
            print("    3. Use 'lance checkpoint' to repair (requires pylance)")
            return None, [missing_path] if missing_path else ["corrupted"]

    if df is not None and len(df) > 0:
        if reported_count and len(df) != reported_count:
            print(f"  WARNING: Only recovered {len(df)} of {reported_count} entries")
        print(f"  First row text preview: {df.iloc[0]['text'][:80]}...")

    return df, errors


def prepare_data_for_new_db(df) -> list[dict]:
    """Prepare data records for insertion into new database."""
    records = []

    for idx, row in df.iterrows():
        record = {
            "text": row.get("text", ""),
            "source_type": row.get("source_type", "session"),
            "source_path": row.get("source_path"),
            "timestamp": row.get("timestamp"),
            "content_hash": row.get("content_hash"),
            "metadata_json": row.get("metadata_json"),
        }

        # Handle timestamp conversion
        ts = record["timestamp"]
        if ts is None:
            record["timestamp"] = datetime.datetime.now()
        elif isinstance(ts, str):
            try:
                ts_clean = ts.replace("Z", "+00:00")
                record["timestamp"] = datetime.datetime.fromisoformat(ts_clean)
            except (ValueError, TypeError):
                record["timestamp"] = datetime.datetime.now()
        elif isinstance(ts, (int, float)):
            try:
                record["timestamp"] = datetime.datetime.fromtimestamp(ts)
            except (ValueError, TypeError):
                record["timestamp"] = datetime.datetime.now()
        elif not isinstance(ts, datetime.datetime):
            record["timestamp"] = datetime.datetime.now()

        records.append(record)

    return records


def create_new_table(db_path: str, embed_fn):
    """Create new table with MiniLM embedding function."""

    class MemorySchema(LanceModel):
        text: str = embed_fn.SourceField()
        vector: Vector(NEW_DIMS) = embed_fn.VectorField()
        source_type: str = "session"
        source_path: str = None  # type: ignore
        timestamp: datetime.datetime = None  # type: ignore
        content_hash: str = None  # type: ignore
        metadata_json: str = None  # type: ignore

    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)

    # Drop existing table if present
    tables = _get_tables_list(db)
    if "memories" in tables:
        print(f"  Dropping existing 'memories' table in {db_path}")
        db.drop_table("memories")

    table = db.create_table("memories", schema=MemorySchema)
    print(f"  Created new table with {NEW_MODEL} ({NEW_DIMS} dims)")
    return table


def migrate_data(records: list[dict], table, embed_fn, device: str):
    """Migrate records to new table with re-embedding."""
    total = len(records)
    print(f"\nMigrating {total} entries with {device} embedding...")
    print(f"  Using batch size: {BATCH_SIZE}")

    success_count = 0
    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            table.add(batch)
            success_count += len(batch)
        except Exception as e:
            print(f"  Batch {batch_num} error: {e}")

        progress = min(i + BATCH_SIZE, total)
        print(
            f"  Progress: {progress}/{total} ({100 * progress / total:.1f}%) - batch {batch_num}/{total_batches}"
        )

    print(f"\n  Migration complete! ({success_count}/{total} entries)")
    return success_count


def print_summary(
    old_count: int,
    new_count: int,
    old_path: str,
    new_path: str,
    device: str,
    errors: list,
):
    """Print migration summary."""
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"  Entries migrated:   {new_count}/{old_count}")
    print(f"  Old DB path:       {old_path}")
    print(f"  New DB path:       {new_path}")
    print(f"  Embedding model:   {NEW_MODEL}")
    print(f"  Device used:       {device}")
    if errors:
        print(f"  Warnings:         {len(errors)} corrupted fragments")
        for err in errors[:3]:
            print(f"    - {err[:80]}...")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate LanceDB from BGE-large-en-v1.5 to all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count entries without writing to new database",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LanceDB Migration: BGE-large → MiniLM")
    print("=" * 60)
    print(f"  Old DB: {OLD_DB_PATH}")
    print(f"  New DB: {NEW_DB_PATH}")
    print(f"  Model:  {NEW_MODEL} ({NEW_DIMS} dims)")

    # Detect device
    device = detect_device()
    print(f"  Device: {device}")

    # Count entries in old DB
    try:
        count, versions = count_old_entries(OLD_DB_PATH)
    except Exception as e:
        print(f"\nError accessing old database: {e}")
        sys.exit(1)

    if args.dry_run:
        print(f"\n[DRY-RUN] Would migrate {count} entries")
        if versions:
            print(f"  Database versions: {len(versions)}")
        return

    if count == 0:
        print("\nNo entries to migrate. Exiting.")
        return

    # Read old data
    errors = []
    try:
        df, errors = read_old_data(OLD_DB_PATH)
    except Exception as e:
        print(f"\nError reading old database: {e}")
        sys.exit(1)

    if df is None or len(df) == 0:
        print("\nNo data to migrate. Exiting.")
        return

    # Prepare records
    print("\nPreparing records for new database...")
    records = prepare_data_for_new_db(df)
    print(f"  Prepared {len(records)} records")

    # Initialize embedding function
    print(f"\nInitializing embedding function on {device}...")
    try:
        embed_fn = get_embedding_function(device)
        print(f"  Model loaded: {NEW_MODEL}")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        print("  Falling back to CPU...")
        device = "cpu"
        embed_fn = get_embedding_function(device)

    # Create new table
    try:
        print(f"\nCreating new database at {NEW_DB_PATH}...")
        table = create_new_table(NEW_DB_PATH, embed_fn)
    except Exception as e:
        print(f"\nError creating new database: {e}")
        sys.exit(1)

    # Migrate data
    try:
        new_count = migrate_data(records, table, embed_fn, device)
    except Exception as e:
        print(f"\nError during migration: {e}")
        sys.exit(1)

    # Print summary
    print_summary(count, new_count, OLD_DB_PATH, NEW_DB_PATH, device, errors)


if __name__ == "__main__":
    main()
