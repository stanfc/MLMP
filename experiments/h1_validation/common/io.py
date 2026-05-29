"""CSV append + resume helpers shared by both experiments."""
from __future__ import annotations
import csv
import os
from typing import Any


def append_row(csv_path: str, row: dict[str, Any]) -> None:
    """Append one row to csv_path. Writes header from row.keys() on first call.

    If the file exists, its header must include all keys in row (we tolerate
    extra columns in the file but not missing ones).
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path)
    if not new_file:
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f))
        missing = set(row.keys()) - set(existing_header)
        if missing:
            raise ValueError(
                f"row has columns not in existing header {existing_header}: {missing}"
            )
        fieldnames = existing_header
    else:
        fieldnames = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def already_done(csv_path: str, key_cols: dict[str, Any]) -> bool:
    """Return True if csv_path contains a row whose values for key_cols all
    match (string-compared, since CSV is text-only).
    """
    if not os.path.exists(csv_path):
        return False
    str_keys = {k: str(v) for k, v in key_cols.items()}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if all(r.get(k, "") == v for k, v in str_keys.items()):
                return True
    return False
