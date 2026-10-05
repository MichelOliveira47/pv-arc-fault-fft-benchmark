"""Validate the labeled CSV input without importing the ML dependencies."""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path


EXPECTED_COLUMNS = ["CH1", "CH2", "CLASSIFIER"]
EXPECTED_LABELS = {"Normal", "Arc"}
EXPERIMENTS = range(1, 17)


def accepted_names(index: int) -> tuple[str, str]:
    return f"Experiment_{index}.csv", f"Experiment_{index:02d}.csv"


def validate_header_and_labels(stream: io.TextIOBase, display_name: str) -> list[str]:
    errors: list[str] = []
    reader = csv.DictReader(stream)
    if reader.fieldnames != EXPECTED_COLUMNS:
        errors.append(
            f"{display_name}: expected header {EXPECTED_COLUMNS}, found {reader.fieldnames}"
        )
        return errors

    observed: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        label = (row.get("CLASSIFIER") or "").strip()
        observed.add(label)
        if label not in EXPECTED_LABELS:
            errors.append(f"{display_name}:{row_number}: invalid CLASSIFIER value {label!r}")
            if len(errors) >= 10:
                break
    if not observed:
        errors.append(f"{display_name}: file has no data rows")
    return errors


def validate_directory(data_dir: Path) -> list[str]:
    errors: list[str] = []
    for index in EXPERIMENTS:
        candidates = [data_dir / name for name in accepted_names(index)]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            errors.append(
                f"Experiment {index}: expected exactly one accepted file, found {len(existing)}"
            )
            continue
        with existing[0].open("r", encoding="utf-8-sig", newline="") as stream:
            errors.extend(validate_header_and_labels(stream, str(existing[0])))
    return errors


def validate_zip(zip_path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        files = {name.replace("\\", "/"): name for name in archive.namelist() if not name.endswith("/")}
        for index in EXPERIMENTS:
            accepted = accepted_names(index)
            matches = [
                original
                for normalized, original in files.items()
                if Path(normalized).name in accepted
            ]
            if len(matches) != 1:
                errors.append(
                    f"Experiment {index}: expected exactly one accepted ZIP member, found {len(matches)}"
                )
                continue
            with archive.open(matches[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                    errors.extend(validate_header_and_labels(stream, matches[0]))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data_labeled/ or data_labeled.zip (default: current directory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    data_dir = base_dir / "data_labeled"
    zip_path = base_dir / "data_labeled.zip"

    if data_dir.is_dir():
        source = data_dir
        errors = validate_directory(data_dir)
    elif zip_path.is_file():
        source = zip_path
        errors = validate_zip(zip_path)
    else:
        print(
            f"No labeled input found. Expected {data_dir} or {zip_path}.",
            file=sys.stderr,
        )
        return 2

    if errors:
        print(f"Validation failed for {source}:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Validated 16 labeled experiment files in {source}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
