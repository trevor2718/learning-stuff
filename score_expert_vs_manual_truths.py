#!/usr/bin/env python3
"""
Compare expert merged extractions (``pdf_*_expert_evaluation.json``) to manual
reference JSON in ``manual_truths/`` using the same cosine-similarity pipeline
as ``SimilarityScorer`` (sentence-transformers).

Manual truth files use the same layout as ``ground_truth/``:

- ``manual_truths/{stem}.json`` (e.g. ``pdf_3.json``), or
- ``manual_truths/mapping.json`` mapping stems to ``paper_N.json``.

Each JSON should validate as ``ResearchPaperExtraction`` (optional ``_meta`` is stripped).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config
from models.schemas import ExpertEvaluation, ResearchPaperExtraction
from models.similarity_scorer import (
    SimilarityScorer,
    load_research_paper_truth,
    resolve_truth_json_path,
)


def _stem_from_expert_filename(path: Path) -> str | None:
    """``pdf_7_expert_evaluation.json`` → ``pdf_7``."""
    name = path.name
    if not name.endswith("_expert_evaluation.json"):
        return None
    return name[: -len("_expert_evaluation.json")]


def _parse_stem_filter(s: str) -> str:
    s = s.strip()
    if s.lower().endswith(".pdf"):
        return Path(s).stem
    return s


def _training_pdf_path(stem: str, pdf_dir: Path | None) -> str:
    if pdf_dir is None:
        return ""
    p = pdf_dir / f"{stem}.pdf"
    return str(p.resolve()) if p.is_file() else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help=f"Directory with pdf_*_expert_evaluation.json (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--manual-truths",
        type=Path,
        default=config.BASE_DIR / "manual_truths",
        help="Folder with manual reference JSON (default: <project>/manual_truths)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.SIMILARITY_SCORE_DIR,
        help=f"Write per-stem JSON here (default: {config.SIMILARITY_SCORE_DIR})",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="If set, fill paper_path when training_papers/{stem}.pdf exists.",
    )
    parser.add_argument(
        "--stem",
        action="append",
        default=None,
        metavar="STEM",
        help="Only score this stem (repeatable), e.g. pdf_3 or pdf_3.pdf",
    )
    parser.add_argument(
        "--encoder-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model id (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    expert_dir: Path = args.expert_dir
    manual_dir: Path = args.manual_truths
    out_dir: Path = args.output_dir

    if not expert_dir.is_dir():
        print(f"Expert directory not found: {expert_dir}", file=sys.stderr)
        sys.exit(2)
    if not manual_dir.is_dir():
        print(f"Manual truths directory not found: {manual_dir}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(expert_dir.glob("pdf_*_expert_evaluation.json"))
    if args.stem is not None:
        want = {_parse_stem_filter(x) for x in args.stem}
        all_files = [
            p for p in all_files if (st := _stem_from_expert_filename(p)) and st in want
        ]

    if not all_files:
        print(f"No pdf_*_expert_evaluation.json files in {expert_dir}", file=sys.stderr)
        sys.exit(1)

    scorer = SimilarityScorer(model_name=args.encoder_model)

    rows: list[dict] = []
    missing_truth: list[str] = []
    errors: list[str] = []

    pdf_dir = Path(args.pdf_dir).resolve() if args.pdf_dir else None

    for expert_path in all_files:
        stem = _stem_from_expert_filename(expert_path)
        if not stem:
            continue

        truth_path = resolve_truth_json_path(stem, manual_dir)
        if truth_path is None:
            missing_truth.append(stem)
            continue

        try:
            with open(expert_path, "r", encoding="utf-8") as f:
                expert_raw = json.load(f)
            evaluation = ExpertEvaluation.model_validate(expert_raw)
        except Exception as e:
            errors.append(f"{expert_path.name}: expert JSON — {e}")
            continue

        gt = load_research_paper_truth(truth_path)
        if gt is None:
            errors.append(f"{stem}: failed to load manual truth {truth_path.name}")
            continue

        try:
            consensus = ResearchPaperExtraction.model_validate(
                {
                    **evaluation.best_extraction.model_dump(mode="json"),
                    "field_confidence": None,
                }
            )
        except Exception as e:
            errors.append(f"{stem}: consensus — {e}")
            continue

        paper_path = _training_pdf_path(stem, pdf_dir)
        sim = scorer.score(consensus, gt, paper_path=paper_path)

        payload = {
            "stem": stem,
            "expert_evaluation_path": str(expert_path.resolve()),
            "manual_truth_path": str(truth_path.resolve()),
            "paper_path": paper_path or None,
            "encoder_model": args.encoder_model,
            "expert_reasoning": evaluation.reasoning,
            "expert_agreement_score": evaluation.agreement_score,
            "expert_field_level_decisions": evaluation.field_level_decisions,
            "expert_best_extraction": evaluation.best_extraction.model_dump(mode="json"),
            "similarity_vs_manual_truth": sim.model_dump(mode="json"),
        }

        out_path = out_dir / f"{stem}_expert_manual_similarity.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        rows.append(
            {
                "stem": stem,
                "mean_overall": sim.mean_overall_similarity,
                "mean_string": sim.mean_string_similarity,
                "mean_specific": sim.mean_specific_similarity,
                "fields_passing": sim.fields_passing,
                "fields_total": sim.fields_total,
                "output": str(out_path),
            }
        )

        print(
            f"{stem}: overall={sim.mean_overall_similarity}  "
            f"passing={sim.fields_passing}/{sim.fields_total}  → {out_path.name}"
        )

    if missing_truth:
        print("\nNo manual truth file for stems:", ", ".join(sorted(missing_truth)), file=sys.stderr)

    if errors:
        print("\nErrors:", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)

    if rows:
        means = [r["mean_overall"] for r in rows if r["mean_overall"] is not None]
        if means:
            avg = sum(means) / len(means)
            print(f"\nScored {len(rows)} paper(s). Mean overall similarity: {avg:.4f}")
        summary_path = out_dir / "expert_manual_similarity_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "expert_dir": str(expert_dir.resolve()),
                    "manual_truths_dir": str(manual_dir.resolve()),
                    "encoder_model": args.encoder_model,
                    "count_scored": len(rows),
                    "mean_overall_similarity": round(sum(means) / len(means), 4) if means else None,
                    "per_stem": rows,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        print(f"Summary: {summary_path}")

    if errors:
        sys.exit(1)
    if not rows and all_files:
        sys.exit(1)


if __name__ == "__main__":
    main()
