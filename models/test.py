"""
Anthropic Claude-based feature extraction.

Uses messages.create() with tool_use to force structured JSON output —
Claude is required to call a defined tool, so the response is always a
validated dict matching the tool's input_schema. No regex/JSON parsing
gymnastics needed.

Two public entry-points:
  • extract_essential_features(file_path)  – low-token, first-page-only pass
  • extract_features(file_path)            – full ResearchPaperExtraction pass
                                             with validated PubMed hyperlink
"""

import time
import base64
import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic

from config import config
from models.schemas import (
    TestPaperExtraction,
    TestPaperConfidence,
    ResearchPaperExtraction,
    ResearchPaperConfidence,
    ModelResponse,
)
from utils.logging_config import logger
from utils.prompts_utils import load_prompt, load_unfilled_features_prompt
from utils.pdf_utils import get_pages
from models.pubmed import PubMedAPI


# ---------------------------------------------------------------------------
# Tool schemas
#
# Defining these as tools and setting tool_choice={"type": "tool", "name": ...}
# forces Claude to always call the tool rather than reply in free text.
# The input_schema acts as the JSON schema Claude must conform to.
# ---------------------------------------------------------------------------

ESSENTIAL_TOOL = {
    "name": "extract_essential_features",
    "description": (
        "Extract essential metadata from the first page of a research paper. "
        "Assign a field_confidence float (0.0–1.0) to every field: "
        "1.0=explicitly stated, 0.7=strongly implied, 0.4=best guess, 0.0=not found."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title of the paper"},
            "authors": {
                "type": "string",
                "description": "Authors formatted as: Last FI, Last FI, ...",
            },
            "year_of_publication": {
                "type": "integer",
                "description": "Year the paper was published",
            },
            "doi": {
                "type": ["string", "null"],
                "description": "DOI string if present, otherwise null",
            },
            "field_confidence": {
                "type": "object",
                "description": "Per-field confidence scores (0.0–1.0)",
                "properties": {
                    "title":               {"type": "number"},
                    "authors":             {"type": "number"},
                    "year_of_publication": {"type": "number"},
                    "doi":                 {"type": "number"},
                },
                "required": ["title", "authors", "year_of_publication", "doi"],
            },
        },
        "required": ["title", "authors", "year_of_publication", "doi", "field_confidence"],
    },
}

FULL_TOOL = {
    "name": "extract_full_features",
    "description": (
        "Extract all structured fields from a full research paper PDF. "
        "Leave pubmed_hyperlink as an empty string — it is resolved separately. "
        "Assign a field_confidence float (0.0–1.0) to every field: "
        "1.0=explicitly stated, 0.7=strongly implied, 0.4=best guess, 0.0=not found. "
        "Set field_confidence.pubmed_hyperlink to 0.0."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "authors":         {"type": "string"},
            "title":           {"type": "string"},
            "journal_name":    {"type": "string"},
            "volume_issue_pages": {"type": "string"},
            "year_of_publication": {"type": "integer"},
            "pubmed_hyperlink": {
                "type": "string",
                "description": "Leave as empty string — filled after tool call",
            },
            "study_design":    {"type": "string"},
            "study_question":  {"type": "string"},
            "population_assessed": {"type": "string"},
            "follow_up_duration":  {"type": "string"},
            "outcome_measures":    {"type": "string"},
            "inclusion_criteria":  {"type": "string"},
            "number_of_participants_in_study": {"type": "integer"},
            "answer_to_study_question": {"type": "string"},
            "field_confidence": {
                "type": "object",
                "description": "Per-field confidence scores (0.0–1.0)",
                "properties": {
                    "authors":             {"type": "number"},
                    "title":               {"type": "number"},
                    "journal_name":        {"type": "number"},
                    "volume_issue_pages":  {"type": "number"},
                    "year_of_publication": {"type": "number"},
                    "pubmed_hyperlink":    {"type": "number"},
                    "study_design":        {"type": "number"},
                    "study_question":      {"type": "number"},
                    "population_assessed": {"type": "number"},
                    "follow_up_duration":  {"type": "number"},
                    "outcome_measures":    {"type": "number"},
                    "inclusion_criteria":  {"type": "number"},
                    "number_of_participants_in_study": {"type": "number"},
                    "answer_to_study_question": {"type": "number"},
                },
                "required": [
                    "authors", "title", "journal_name", "volume_issue_pages",
                    "year_of_publication", "pubmed_hyperlink", "study_design",
                    "study_question", "population_assessed", "follow_up_duration",
                    "outcome_measures", "inclusion_criteria",
                    "number_of_participants_in_study", "answer_to_study_question",
                ],
            },
        },
        "required": [
            "authors", "title", "journal_name", "volume_issue_pages",
            "year_of_publication", "pubmed_hyperlink", "study_design",
            "study_question", "population_assessed", "follow_up_duration",
            "outcome_measures", "inclusion_criteria",
            "number_of_participants_in_study", "answer_to_study_question",
            "field_confidence",
        ],
    },
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ESSENTIAL_SYSTEM = (
    "You are a precise academic-paper metadata extractor. "
    "Always call the provided tool with the extracted values — never reply in plain text."
)

FULL_SYSTEM = (
    "You are a precise clinical/academic paper data extractor. "
    "Always call the provided tool with the extracted values — never reply in plain text."
)


def _get_tool_input(response) -> dict:
    """Pull the tool-use input block out of a messages.create() response."""
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError(
        f"Claude did not call the tool. stop_reason={response.stop_reason}, "
        f"content={response.content}"
    )


class ClaudeExtractor:
    """Handles feature extraction using Anthropic's Claude models."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.client = Anthropic(api_key=self.api_key)
        self.model = config.ANTHROPIC_MODEL
        self.temperature = config.DEFAULT_TEMPERATURE
        self.name = "ClaudeExtractor"
        logger.info("Claude extractor initialized.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pdf_to_base64(self, path: Path) -> str:
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _validate_connection(self) -> bool:
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            logger.info(f"Claude API connection successful. model={self.model}")
            return True
        except Exception as e:
            logger.error(f"Claude connection failed: {e}")
            return False

    # ------------------------------------------------------------------
    # 1. Low-token essential-features pass  (first page only)
    # ------------------------------------------------------------------

    def extract_essential_features(self, file_path: Path) -> TestPaperExtraction:
        """
        Low-token pass: reads only page 1 of the PDF.

        Forces Claude to call the extract_essential_features tool via
        tool_choice={"type": "tool", "name": "extract_essential_features"},
        so block.input is always a schema-conformant dict — no JSON parsing needed.
        """
        logger.debug("Claude: extracting essential features (page 1 only)")

        first_page_pdf = get_pages(file_path, page_numbers=[0])
        pdf_b64 = self._pdf_to_base64(first_page_pdf)

        try:
            user_text = load_prompt(config.ANTHROPIC_ESSENTIAL_FEATURES_PROMPT)
        except Exception:
            user_text = "Extract the essential metadata from this research paper."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=ESSENTIAL_SYSTEM,
            tools=[ESSENTIAL_TOOL],
            tool_choice={"type": "tool", "name": "extract_essential_features"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )

        data = _get_tool_input(response)
        logger.debug(f"Claude essential tool input: {data}")

        confidence_data = data.pop("field_confidence", {})
        features = TestPaperExtraction(**data)
        features.field_confidence = TestPaperConfidence(**confidence_data)

        print("\n=== Claude TestPaperExtraction ===")
        print(features.model_dump_json(indent=2))
        return features

    # ------------------------------------------------------------------
    # 2. Full extraction pass
    # ------------------------------------------------------------------

    def extract_all_features(
        self,
        file_path: Path,
        essential_features: Optional[TestPaperExtraction] = None,
    ) -> ResearchPaperExtraction:
        """
        Full pass over the complete PDF.

        Forces Claude to call the extract_full_features tool. The returned
        pubmed_hyperlink placeholder is replaced with a validated URL from
        the PubMed API (up to 5 progressive search attempts).
        """
        logger.debug("Claude: extracting all features")

        pdf_b64 = self._pdf_to_base64(file_path)

        try:
            user_text = load_unfilled_features_prompt(
                essential_features, config.ANTHROPIC_FEATURES_PROMPT
            )
        except Exception:
            user_text = "Extract all structured fields from this research paper."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=FULL_SYSTEM,
            tools=[FULL_TOOL],
            tool_choice={"type": "tool", "name": "extract_full_features"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )

        data = _get_tool_input(response)
        logger.debug(f"Claude full tool input: {data}")

        confidence_data = data.pop("field_confidence", {})

        # Resolve the PubMed link and patch it in
        pubmed_link = self._resolve_pubmed_link(essential_features)
        data["pubmed_hyperlink"] = pubmed_link or "https://pubmed.ncbi.nlm.nih.gov/"

        extraction = ResearchPaperExtraction(**data)
        extraction.field_confidence = ResearchPaperConfidence(**confidence_data)
        extraction.field_confidence.pubmed_hyperlink = 1.0 if pubmed_link else 0.0

        return extraction

    def _resolve_pubmed_link(
        self,
        essential_features: Optional[TestPaperExtraction],
        max_attempts: int = 5,
    ) -> Optional[str]:
        """
        Query PubMed with progressively relaxed criteria until a valid link is
        found or max_attempts is exhausted.

        Strategy ladder (most → least specific):
          1. DOI + title + first author + year
          2. title + first author + year
          3. title + first author
          4. title only
          5. DOI only (skipped if doi is None)
        """
        if essential_features is None:
            return None

        pubmed = PubMedAPI()
        authors = essential_features.authors
        title   = essential_features.title
        year    = essential_features.year_of_publication
        doi     = essential_features.doi

        strategies = [
            dict(authors=authors, title=title, year=year, doi=doi),
            dict(authors=authors, title=title, year=year, doi=None),
            dict(authors=authors, title=title, year=None, doi=None),
            dict(authors=None,    title=title, year=None, doi=None),
            dict(authors=None,    title=None,  year=None, doi=doi) if doi else None,
        ]

        for idx, strategy in enumerate(strategies[:max_attempts]):
            if strategy is None:
                continue
            try:
                pmid = pubmed.find_pmid_by_metadata(**strategy)
                if pmid:
                    link = pubmed.get_pubmed_link_by_pmid(pmid)
                    if link:
                        logger.info(f"PubMed link resolved on attempt {idx+1}: {link}")
                        return link
            except Exception as e:
                logger.warning(f"PubMed attempt {idx+1} failed: {e}")

        logger.warning("Could not resolve PubMed link after all attempts.")
        return None

    # ------------------------------------------------------------------
    # Pipeline entry-point
    # ------------------------------------------------------------------

    def extract_features(self, file_path: Path) -> ModelResponse:
        """Full pipeline: essential → PubMed → full extraction."""
        start = time.time()
        try:
            essential  = self.extract_essential_features(file_path)
            extraction = self.extract_all_features(file_path, essential)
            return ModelResponse(
                model_name=self.model,
                provider="Anthropic",
                essential_extraction=essential,
                extraction=extraction,
                processing_time=time.time() - start,
                error=None,
            )
        except Exception as e:
            logger.error(f"Claude extraction failed: {e}")
            return ModelResponse(
                model_name=self.model,
                provider="Anthropic",
                extraction=None,
                processing_time=time.time() - start,
                error=str(e),
            )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pdf = config.TRAINING_DIR / "pdf_1.pdf"
    extractor = ClaudeExtractor()
    # Quick low-token test
    essential = extractor.extract_essential_features(pdf)
    print(essential)
    # Full pipeline
    # result = extractor.extract_features(pdf)
    # print(result.model_dump_json(indent=2))