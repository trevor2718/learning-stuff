#!/usr/bin/env python3
"""
Convert each ``pdf_*_expert_evaluation.json`` under an input directory to a
one-row CSV with the same basename (``pdf_*_expert_evaluation.csv``).

Run from project root:

- Same folder as JSON (default): ``python3 convert_expert_json_to_csv.py``
- Read ``output/``, write CSVs to ``output_expert/``::

    python3 convert_expert_json_to_csv.py --input-dir output --output-dir output_expert
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Matches ResearchPaperExtractionBody field order
EXTRACTION_FIELDS = [
    "authors",
    "title",
    "journal_name",
    "volume_issue_pages",
    "year_of_publication",
    "doi",
    "pubmed_hyperlink",
    "study_design",
    "study_question",
    "population_assessed",
    "follow_up_duration",
    "outcome_measures",
    "inclusion_criteria",
    "number_of_participants_in_study",
    "answer_to_study_question",
]

META_COLUMNS = ["reasoning", "agreement_score", "field_level_decisions_json"]


def _cell(v: object) -> str:
    if v is None:
        return ""
    return str(v)


def json_to_row(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    best = data.get("best_extraction") or {}
    row: dict[str, str] = {"source_file": path.name}
    for key in EXTRACTION_FIELDS:
        row[key] = _cell(best.get(key))
    row["reasoning"] = _cell(data.get("reasoning"))
    row["agreement_score"] = _cell(data.get("agreement_score"))
    fld = data.get("field_level_decisions") or {}
    row["field_level_decisions_json"] = json.dumps(fld, ensure_ascii=False)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "output_expert",
        help="Directory containing pdf_*_expert_evaluation.json (default: output_expert)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV files (default: same as --input-dir)",
    )
    args = parser.parse_args()

    in_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = in_dir
    else:
        out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir

    if not in_dir.is_dir():
        print(f"Missing directory: {in_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(in_dir.glob("pdf_*_expert_evaluation.json"))
    if not paths:
        print(f"No pdf_*_expert_evaluation.json in {in_dir}", file=sys.stderr)
        sys.exit(1)

    header = ["source_file"] + EXTRACTION_FIELDS + META_COLUMNS

    for path in paths:
        row = json_to_row(path)
        csv_path = out_dir / path.with_suffix(".csv").name
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            w.writerow(row)
        print(csv_path.name)

    print(f"Wrote {len(paths)} CSV file(s) in {out_dir}")


if __name__ == "__main__":
    main()
