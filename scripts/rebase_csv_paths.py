#!/usr/bin/env python3
"""
rebase_csv_paths.py

Replaces the old base path with a new base path in all CSV files found
in a given directory. Only path columns are modified:
  - frame_path
  - video_path
  - subject_dir
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Columns that contain filesystem paths
PATH_COLUMNS = {"frame_path", "video_path", "subject_dir"}


def rebase_path(value: str, old_base: str, new_base: str) -> str:
    """Replace old_base prefix with new_base in a path string."""
    if value.startswith(old_base):
        return new_base + value[len(old_base):]
    return value


def process_csv(
    csv_path: Path,
    old_base: str,
    new_base: str,
    dry_run: bool = False,
) -> int:
    """
    Process a single CSV file.
    Returns the number of cells modified.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"  [SKIP] {csv_path.name} — no header found")
            return 0

        fieldnames = list(reader.fieldnames)
        active_cols = PATH_COLUMNS & set(fieldnames)

        if not active_cols:
            print(f"  [SKIP] {csv_path.name} — no path columns found")
            return 0

        rows = list(reader)

    modified = 0
    for row in rows:
        for col in active_cols:
            original = row[col]
            updated = rebase_path(original, old_base, new_base)
            if updated != original:
                row[col] = updated
                modified += 1

    if dry_run:
        print(f"  [DRY-RUN] {csv_path.name} — {modified} cells would change")
        return modified

    # Write back to the same file
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [DONE] {csv_path.name} — {modified} cells updated")
    return modified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebase filesystem paths in dataset CSV files."
    )
    parser.add_argument(
        "csv_dir",
        type=str,
        help="Directory containing the CSV files to process.",
    )
    parser.add_argument(
        "old_base",
        type=str,
        help="Old base path prefix to replace (e.g. /home/shahriar/Documents/).",
    )
    parser.add_argument(
        "new_base",
        type=str,
        help="New base path prefix to use (e.g. /home/name/Projects/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to disk.",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_dir():
        print(f"Error: '{csv_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Normalize bases: ensure they end with the OS separator
    old_base = args.old_base.rstrip("/") + "/"
    new_base = args.new_base.rstrip("/") + "/"

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the directory.")
        sys.exit(0)

    print(f"Directory : {csv_dir}")
    print(f"Old base  : {old_base}")
    print(f"New base  : {new_base}")
    print(f"Dry run   : {args.dry_run}")
    print(f"Files     : {len(csv_files)}\n")

    total_modified = 0
    for csv_path in csv_files:
        total_modified += process_csv(csv_path, old_base, new_base, args.dry_run)

    print(f"\nTotal cells {'(would be) ' if args.dry_run else ''}modified: {total_modified}")


if __name__ == "__main__":
    main()
