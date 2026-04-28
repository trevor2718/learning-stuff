#!/usr/bin/env python3
"""
Load saved per-model extractions from ``output/``, merge with the expert prompt via
**OpenRouter** (default: ``OPENROUTER_EXPERT_MODEL``, e.g. GPT-5.4).

With no ``stem`` argument: processes every ``training_papers/*.pdf`` whose
``output/{stem}_expert_evaluation.json`` does not exist yet.

With a ``stem`` argument: processes that stem only (overwrites expert evaluation if present).

Expected input files (for each ``stem``):

- ``{stem}_claude.json``
- ``{stem}_openai_openrouterapi.json`` **or** ``{stem}_openrouter.json``
- ``{stem}_gemini.json``

Output:

- ``output/{stem}_expert_evaluation.json`` — full ``ExpertEvaluation`` from OpenRouter

(Optional similarity vs ground truth is left commented in the script body.)

Environment: ``OPENROUTER_API_KEY`` (required), optional ``OPENROUTER_EXPERT_MODEL``.
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
from models.expert_evaluator import evaluate_extractions_openrouter, load_model_outputs_triplet
from utils.logging_config import logger


def _normalize_stem(stem: str) -> str:
    s = stem.strip()
    if s.lower().endswith(".pdf"):
        return Path(s).stem
    return s


def _pdf_number(stem: str) -> int:
    """Extract trailing integer from a stem like 'pdf_7' → 7 (non-numeric stems → -1)."""
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def _stems_to_process(
    stem_arg: str | None,
    training_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
    start: int | None = None,
    end: int | None = None,
) -> list[str]:
    """Stems needing an expert evaluation.

    Batch mode (no stem_arg): iterate every training_papers/*.pdf sorted
    numerically, optionally filtered to [start, end] by PDF number, and
    optionally skipping stems that already have an expert evaluation (unless
    force=True).

    Single-stem mode: return that one stem only.
    """
    if stem_arg is not None:
        one = _normalize_stem(stem_arg)
        return [one] if one else []

    all_pdfs = sorted(training_dir.glob("*.pdf"), key=lambda p: _pdf_number(p.stem))
    result: list[str] = []
    for pdf in all_pdfs:
        st = pdf.stem
        num = _pdf_number(st)
        if start is not None and num < start:
            continue
        if end is not None and num > end:
            continue
        if not force and (output_dir / f"{st}_expert_evaluation.json").exists():
            continue
        result.append(st)
    return result


def _run_expert_for_stem(
    stem: str,
    output_dir: Path,
    expert_model: str | None,
) -> bool:
    """
    Run OpenRouter expert and write ``{stem}_expert_evaluation.json``.
    Returns True if written, False if skipped (no model outputs).
    """
    responses = load_model_outputs_triplet(output_dir, stem)
    if not responses:
        print(
            f"Skipping {stem!r}: no usable model outputs in {output_dir}.",
            file=sys.stderr,
        )
        return False
    if len(responses) < 3:
        logger.warning(
            "Expected 3 files (claude, openrouter/openai_openrouterapi, gemini); "
            f"only {len(responses)} loaded for {stem!r}."
        )

    print(f"[{stem}] Loaded {len(responses)} model output(s); running OpenRouter expert…")
    evaluation = evaluate_extractions_openrouter(responses, model=expert_model)

    expert_eval_path = output_dir / f"{stem}_expert_evaluation.json"
    with open(expert_eval_path, "w", encoding="utf-8") as f:
        json.dump(
            evaluation.model_dump(mode="json"),
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    print(f"Wrote {expert_eval_path}")

    # expert_fields = evaluation.best_extraction.model_dump(mode="json")

    # gt = load_ground_truth(pdf_path)
    # if gt is None:
    #     print(
    #         f"No ground truth JSON resolved for {pdf_path} (stem={pdf_path.stem!r}).",
    #         file=sys.stderr,
    #     )
    #     sys.exit(1)

    # consensus = ResearchPaperExtraction.model_validate(
    #     {**expert_fields, "field_confidence": None}
    # )
    # scorer = SimilarityScorer()
    # sim = scorer.score(consensus, gt, paper_path=str(pdf_path))

    # expert_model = args.expert_model or config.OPENROUTER_EXPERT_MODEL

    # payload = {
    #     "stem": stem,
    #     "pdf_path": str(pdf_path.resolve()),
    #     "openrouter_expert_model": expert_model,
    #     "source_outputs": {
    #         "output_dir": str(output_dir.resolve()),
    #         "files": [
    #             f"{stem}_claude.json",
    #             f"{stem}_openai_openrouterapi.json or {stem}_openrouter.json",
    #             f"{stem}_gemini.json",
    #         ],
    #     },
    #     "expert_reasoning": evaluation.reasoning,
    #     "expert_agreement_score": evaluation.agreement_score,
    #     "expert_field_level_decisions": evaluation.field_level_decisions,
    #     "expert_best_extraction": expert_fields,
    #     "similarity_vs_ground_truth": sim.model_dump(mode="json"),
    # }

    # out_path = similarity_dir / f"{stem}_expert_similarity.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    # print(f"Wrote {out_path}")
    # print(
    #     f"Mean overall similarity: {sim.mean_overall_similarity}  "
    #     f"(fields passing {sim.fields_passing}/{sim.fields_total})"
    # )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stem",
        nargs="?",
        default=None,
        help=(
            "PDF stem (e.g. pdf_1). If omitted, process every training_papers/*.pdf "
            "that lacks output/{stem}_expert_evaluation.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR),
        help="Directory with per-model JSON (default: project output/)",
    )
    parser.add_argument(
        "--expert-model",
        default=None,
        help="OpenRouter model id (overrides OPENROUTER_EXPERT_MODEL)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run expert evaluation even if output already exists.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        metavar="N",
        help="First PDF number to process in batch mode (inclusive, e.g. --start 5).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        metavar="N",
        help="Last PDF number to process in batch mode (inclusive, e.g. --end 15).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = config.TRAINING_DIR

    if not config.OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    stems = _stems_to_process(
        args.stem,
        training_dir,
        output_dir,
        force=args.force,
        start=args.start,
        end=args.end,
    )

    if not stems:
        if args.stem is None:
            print(
                "Nothing to do: every PDF in training_papers already has "
                f"{output_dir.name}/{{stem}}_expert_evaluation.json "
                "(or training_papers is empty). Use --force to reprocess."
            )
        else:
            print(
                f"Invalid or empty stem {args.stem!r}; provide e.g. pdf_1.",
                file=sys.stderr,
            )
            sys.exit(2)
        sys.exit(0)

    ok = 0
    for stem in stems:
        if _run_expert_for_stem(stem, output_dir, args.expert_model):
            ok += 1

    print(f"Done: wrote {ok}/{len(stems)} expert evaluation(s).")
    if ok < len(stems):
        sys.exit(1)


if __name__ == "__main__":
    main()
