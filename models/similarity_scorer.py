"""
Cosine similarity scorer using sentence-transformers (BERT).
Compares a consensus extraction against WikiStim ground truth.

Feature categories from the pre-registration:
  - String features  (target >= 0.8): study_question, population_assessed,
    follow_up_duration, answer_to_study_question, inclusion_criteria,
    outcome_measures, number_of_participants_in_study, study_design
  - Specific features (target >= 0.9): authors, title, journal_name,
    volume_issue_pages, year_of_publication, pubmed_hyperlink
"""

import json
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer, util

from config import config
from models.schemas import (
    ResearchPaperExtraction,
    TestPaperExtraction,
    FieldSimilarity,
    SimilarityResult,
)
from utils.logging_config import logger


STRING_FEATURES = [
    "study_question",
    "population_assessed",
    "follow_up_duration",
    "answer_to_study_question",
    "inclusion_criteria",
    "outcome_measures",
    "number_of_participants_in_study",
    "study_design",
]
STRING_TARGET = 0.8

SPECIFIC_FEATURES = [
    "authors",
    "title",
    "journal_name",
    "volume_issue_pages",
    "year_of_publication",
    "pubmed_hyperlink",
]
SPECIFIC_TARGET = 0.9

# CATEGORICAL_FEATURES = [
    
# ]

# Fields present on TestPaperExtraction that overlap ground-truth metadata
ESSENTIAL_TRUTH_FIELDS = ["authors", "title", "year_of_publication"]


class SimilarityScorer:
    """
    Encodes extraction fields and ground truth with a sentence-transformer model,
    then computes per-field cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading sentence-transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        logger.info("Sentence-transformer model loaded.")

    def score(
        self,
        consensus: ResearchPaperExtraction,
        ground_truth: ResearchPaperExtraction,
        paper_path: str = "",
    ) -> SimilarityResult:
        """
        Compare a consensus extraction against ground truth field by field.

        Args:
            consensus: The expert evaluator's consensus extraction
            ground_truth: The WikiStim manually curated extraction
            paper_path: Path to source PDF (for bookkeeping)

        Returns:
            SimilarityResult with per-field cosine similarity scores
        """
        consensus_dict = consensus.model_dump(mode='json', exclude={'field_confidence', 'doi'})
        truth_dict = ground_truth.model_dump(mode='json', exclude={'field_confidence', 'doi'})

        string_results = self._score_fields(consensus_dict, truth_dict, STRING_FEATURES, STRING_TARGET)
        specific_results = self._score_fields(consensus_dict, truth_dict, SPECIFIC_FEATURES, SPECIFIC_TARGET)

        all_scores = [f.cosine_similarity for f in string_results + specific_results]
        string_scores = [f.cosine_similarity for f in string_results]
        specific_scores = [f.cosine_similarity for f in specific_results]

        result = SimilarityResult(
            paper_path=paper_path,
            string_features=string_results,
            specific_features=specific_results,
            mean_string_similarity=_safe_mean(string_scores),
            mean_specific_similarity=_safe_mean(specific_scores),
            mean_overall_similarity=_safe_mean(all_scores),
            fields_passing=sum(1 for f in string_results + specific_results if f.passed),
            fields_total=len(all_scores),
        )

        logger.info(
            f"Similarity: overall={result.mean_overall_similarity:.3f}  "
            f"string={result.mean_string_similarity:.3f}  "
            f"specific={result.mean_specific_similarity:.3f}  "
            f"passing={result.fields_passing}/{result.fields_total}"
        )

        return result

    def score_essential(
        self,
        essential: TestPaperExtraction,
        ground_truth: ResearchPaperExtraction,
        paper_path: str = "",
    ) -> SimilarityResult:
        """
        Compare essential (TestPaperExtraction) fields to the same fields in ground truth.

        Uses the same sentence-encoder cosine similarity as ``score`` but only for
        authors, title, year_of_publication, and optionally doi when either side has it.
        """
        ess = essential.model_dump(mode="json", exclude={"field_confidence"})
        truth_full = ground_truth.model_dump(mode="json", exclude={"field_confidence"})

        consensus_dict = {k: ess.get(k) for k in ESSENTIAL_TRUTH_FIELDS}
        truth_dict = {k: truth_full.get(k) for k in ESSENTIAL_TRUTH_FIELDS}

        specific_results = self._score_fields(
            consensus_dict, truth_dict, ESSENTIAL_TRUTH_FIELDS, SPECIFIC_TARGET
        )

        if ess.get("doi") or truth_full.get("doi"):
            doi_cons = {"doi": ess.get("doi")}
            doi_truth = {"doi": truth_full.get("doi")}
            specific_results.extend(
                self._score_fields(doi_cons, doi_truth, ["doi"], SPECIFIC_TARGET)
            )

        all_scores = [f.cosine_similarity for f in specific_results]

        result = SimilarityResult(
            paper_path=paper_path,
            string_features=[],
            specific_features=specific_results,
            mean_string_similarity=None,
            mean_specific_similarity=_safe_mean(all_scores),
            mean_overall_similarity=_safe_mean(all_scores),
            fields_passing=sum(1 for f in specific_results if f.passed),
            fields_total=len(specific_results),
        )

        logger.info(
            f"Essential-field similarity: mean={result.mean_overall_similarity}  "
            f"passing={result.fields_passing}/{result.fields_total}"
        )

        return result

    def _score_fields(
        self,
        consensus_dict: dict,
        truth_dict: dict,
        fields: list[str],
        target: float,
    ) -> list[FieldSimilarity]:
        """Encode and compare a list of fields."""
        results = []

        for field in fields:
            raw_model = consensus_dict.get(field)
            raw_truth = truth_dict.get(field)
            model_val = str(raw_model) if raw_model is not None else ""
            truth_val = str(raw_truth) if raw_truth is not None else ""

            if not model_val and not truth_val:
                cos_sim = 1.0
            elif not model_val or not truth_val:
                cos_sim = 0.0
            else:
                emb_model = self.encoder.encode(model_val, convert_to_tensor=True)
                emb_truth = self.encoder.encode(truth_val, convert_to_tensor=True)
                cos_sim = util.cos_sim(emb_model, emb_truth).item()

            results.append(FieldSimilarity(
                field_name=field,
                model_value=model_val,
                ground_truth_value=truth_val,
                cosine_similarity=round(cos_sim, 4),
                target=target,
                passed=cos_sim >= target,
            ))

        return results


def resolve_truth_json_path(stem: str, truth_dir: Path) -> Optional[Path]:
    """
    Resolve a curated JSON path for a PDF stem.

    Lookup order:
      1. Direct stem match — {truth_dir}/{stem}.json
      2. Mapping file      — {truth_dir}/mapping.json maps stem → paper number,
                             then {truth_dir}/paper_{number}.json
    """
    direct = truth_dir / f"{stem}.json"
    if direct.exists():
        return direct

    mapping_path = truth_dir / "mapping.json"
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            paper_num = mapping.get(stem)
            if paper_num is not None:
                mapped = truth_dir / f"paper_{paper_num}.json"
                if mapped.exists():
                    return mapped
                logger.warning(
                    f"Mapping points to paper_{paper_num}.json but file not found under {truth_dir}"
                )
        except Exception as e:
            logger.error(f"Failed to read {mapping_path}: {e}")

    return None


def load_research_paper_truth(json_path: Path) -> Optional[ResearchPaperExtraction]:
    """
    Load ``ResearchPaperExtraction`` from a JSON file.

    The ``_meta`` key (analyst info, row source) is stripped before parsing.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.pop("_meta", None)

        return ResearchPaperExtraction(**data)
    except Exception as e:
        logger.error(f"Failed to load research paper truth {json_path}: {e}")
        return None


def load_ground_truth(pdf_path: Path) -> Optional[ResearchPaperExtraction]:
    """
    Load ground truth JSON for a paper from the ground_truth/ directory.

    Supports two naming conventions:
      - Direct match:  ground_truth/pdf_17.json   (same stem as the PDF)
      - Mapped match:  ground_truth/paper_77.json  (via mapping.json)

    The ``_meta`` key (analyst info, row source) is stripped before parsing.
    """
    gt_path = resolve_truth_json_path(pdf_path.stem, config.GROUND_TRUTH_DIR)

    if gt_path is None:
        logger.warning(f"No ground truth found for {pdf_path.name}")
        return None

    return load_research_paper_truth(gt_path)


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)
