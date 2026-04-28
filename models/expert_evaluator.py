"""
Uses OpenAI GPT as expert judge.
"""

import json
import csv
import re
import time
from pathlib import Path
from string import Template
from typing import List, Optional, Tuple

# ---- remove. used for direct testing of API 
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# ----

from openai import OpenAI

from config import config
from models.schemas import (
    ModelResponse,
    ExpertEvaluation,
    ResearchPaperExtraction,
    ResearchPaperExtractionBody,
)
from utils.logging_config import logger
from utils.prompts_utils import load_prompt


def _extract_json_object_from_llm_text(response_text: str) -> dict:
    """
    Parse a JSON object from an LLM reply, tolerating markdown fences and preamble.
    """
    text = response_text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start < 0 or json_end <= json_start:
        raise ValueError("No JSON object found in expert response")
    return json.loads(text[json_start:json_end])


def load_model_responses_from_split_dir(output_dir: Path, stem: str) -> Optional[List[ModelResponse]]:
    """
    Build ModelResponse list from per-model files saved under output_dir:
    {stem}_openai.json, {stem}_claude.json, {stem}_gemini.json
    (each file: ResearchPaperExtraction-shaped JSON).
    """
    mapping = (
        ("_openai.json", "OpenAI", config.OPENAI_MODEL),
        ("_claude.json", "Anthropic", config.ANTHROPIC_MODEL),
        ("_gemini.json", "Google", config.GOOGLE_MODEL),
    )
    out: List[ModelResponse] = []
    for suffix, provider, model_name in mapping:
        path = output_dir / f"{stem}{suffix}"
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            extraction = ResearchPaperExtraction.model_validate(data)
            out.append(
                ModelResponse(
                    model_name=model_name,
                    provider=provider,
                    extraction=extraction,
                    processing_time=None,
                    error=None,
                )
            )
        except Exception as e:
            logger.warning(f"Skipping {path.name}: {e}")
    return out if out else None


class ExpertEvaluator:
    """
    Expert model that evaluates and selects the best extraction from multiple models.
    """

    def __init__(self, api_key: str = None):
        # TODO: update expert evaluator model to something else?
        self.api_key = api_key or config.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.OPENAI_EXPERT_MODEL
        self.name = "ExpertEvaluator"

    # 1. Run all Extractors

    def run_extractors(self, pdf_path: Path) -> List[ModelResponse]:
        """
        Run Claude, Gemini, and OpenAI extractors on a PDF and return their responses.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of ModelResponse objects (one per extractor).
        """
        from models.claude_extractor import ClaudeExtractor
        from models.gemini_extractor import GeminiExtractor
        from models.openai_extractor import OpenAIExtractor

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        claude_extractor = ClaudeExtractor()
        gemini_extractor = GeminiExtractor()
        openai_extractor = OpenAIExtractor()

        logger.info(f"Running Claude extractor on {pdf_path.name}")
        claude_response = claude_extractor.extract_features(pdf_path)

        logger.info(f"Running Gemini extractor on {pdf_path.name}")
        gemini_response = gemini_extractor.extract_features(pdf_path)

        logger.info(f"Running OpenAI extractor on {pdf_path.name}")
        openai_response = openai_extractor.extract_features(pdf_path)

        return [claude_response, gemini_response, openai_response]

    # 2. Expert Evaluate on Outputs

    def evaluate_extractions(self, model_responses: List[ModelResponse]) -> ExpertEvaluation:
        """
        Evaluate multiple model responses and determine the best extraction.
        Each model's full extraction (including field_confidence when present) is
        sent to the expert as JSON; the returned best_extraction omits confidence.

        Args:
            model_responses: List of responses from different models.

        Returns:
            ExpertEvaluation with the best extraction and reasoning.
        """
        logger.info("Starting expert evaluation of model responses")

        successful_responses = [response for response in model_responses if response.extraction is not None]

        if not successful_responses:
            raise ValueError("No successful extractions to evaluate")

        if len(successful_responses) == 1:
            logger.info("Only one successful extraction, using it directly")
            body = self._body_from_extraction(successful_responses[0].extraction)
            return ExpertEvaluation(
                best_extraction=body,
                reasoning="Only one model successfully extracted features.",
                field_level_decisions={},
                agreement_score=None,
            )

        payloads = [
            {
                "provider": r.provider,
                "model_name": r.model_name,
                "extraction": r.extraction.model_dump(mode="json"),
            }
            for r in successful_responses
        ]
        responses_json = json.dumps(payloads, indent=2, ensure_ascii=False)

        raw_prompt = load_prompt(config.OPENAI_EXPERT_PROMPT)
        prompt = Template(raw_prompt).safe_substitute(responses_json=responses_json)

        system_prompt_path = config.OPENAI_EXPERT_SYSTEM_PROMPT
        system_content = (
            load_prompt(system_prompt_path)
            if system_prompt_path and system_prompt_path.exists()
            else "You are an expert at evaluating and comparing scientific paper extractions."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            response_text = (response.choices[0].message.content or "").strip()
            data = _extract_json_object_from_llm_text(response_text)
            return self._parse_expert_judgment_response(data)

        except Exception as e:
            logger.error(f"Expert evaluation failed: {e}")
            logger.warning("Falling back to first successful extraction")
            body = self._body_from_extraction(successful_responses[0].extraction)
            return ExpertEvaluation(
                best_extraction=body,
                reasoning=f"Expert evaluation failed: {e}. Using first successful extraction.",
                field_level_decisions={},
                agreement_score=None,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_expert_judgment_response(data: dict) -> ExpertEvaluation:
        """Parse expert LLM JSON into ExpertEvaluation (no field_confidence on output)."""
        body_keys = set(ResearchPaperExtractionBody.model_fields.keys())

        if isinstance(data.get("best_extraction"), dict):
            best_raw = dict(data["best_extraction"])
            best_raw.pop("field_confidence", None)
            best_extraction = ResearchPaperExtractionBody(**best_raw)
            return ExpertEvaluation(
                best_extraction=best_extraction,
                reasoning=str(data.get("reasoning", "") or ""),
                field_level_decisions=data.get("field_level_decisions") or {},
                agreement_score=ExpertEvaluator._normalize_agreement_score(
                    data.get("agreement_score")
                ),
            )

        if isinstance(data.get("extraction"), dict):
            inner = dict(data["extraction"])
            inner.pop("field_confidence", None)
            best_extraction = ResearchPaperExtractionBody(**inner)
            return ExpertEvaluation(
                best_extraction=best_extraction,
                reasoning=str(data.get("reasoning", "") or ""),
                field_level_decisions=data.get("field_level_decisions") or {},
                agreement_score=ExpertEvaluator._normalize_agreement_score(
                    data.get("agreement_score")
                ),
            )

        flat = dict(data)
        reasoning = str(flat.pop("reasoning", "") or "").strip() or "No reasoning provided."
        agreement_score = ExpertEvaluator._normalize_agreement_score(
            flat.pop("agreement_score", None)
        )
        fld = flat.pop("field_level_decisions", None)
        field_level_decisions = fld if isinstance(fld, dict) else {}

        inner = {k: v for k, v in flat.items() if k in body_keys}
        inner.pop("field_confidence", None)
        best_extraction = ResearchPaperExtractionBody(**inner)

        return ExpertEvaluation(
            best_extraction=best_extraction,
            reasoning=reasoning,
            field_level_decisions=field_level_decisions,
            agreement_score=agreement_score,
        )

    @staticmethod
    def _body_from_extraction(extraction: ResearchPaperExtraction) -> ResearchPaperExtractionBody:
        data = extraction.model_dump(exclude={"field_confidence"}, mode="json")
        return ResearchPaperExtractionBody(**data)

    @staticmethod
    def _normalize_agreement_score(raw: object) -> Optional[float]:
        if raw is None:
            return None
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return None
        if x > 1.0:
            x = x / 100.0
        return max(0.0, min(1.0, x))


def load_model_outputs_triplet(output_dir: Path, stem: str) -> Optional[List[ModelResponse]]:
    """
    Load the three per-model JSON files for one PDF stem (Claude, OpenRouter/OpenAI slot, Gemini).

    Tries ``{stem}_openai_openrouterapi.json`` first, then ``{stem}_openrouter.json``.
    """
    output_dir = Path(output_dir)
    candidates_openrouter = (
        output_dir / f"{stem}_openai_openrouterapi.json",
        output_dir / f"{stem}_openrouter.json",
    )
    openrouter_path = next((p for p in candidates_openrouter if p.is_file()), None)

    mapping: List[Tuple[Path, str, str]] = [
        (output_dir / f"{stem}_claude.json", "Anthropic", config.ANTHROPIC_MODEL),
    ]
    if openrouter_path:
        mapping.append(
            (openrouter_path, "OpenRouter", config.OPENROUTER_MODEL),
        )
    mapping.append(
        (output_dir / f"{stem}_gemini.json", "Google", config.GOOGLE_MODEL),
    )

    out: List[ModelResponse] = []
    for path, provider, model_name in mapping:
        if not path.is_file():
            logger.warning(f"Missing output file: {path.name}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            extraction = ResearchPaperExtraction.model_validate(data)
            out.append(
                ModelResponse(
                    model_name=model_name,
                    provider=provider,
                    extraction=extraction,
                    processing_time=None,
                    error=None,
                )
            )
        except Exception as e:
            logger.warning(f"Skipping {path.name}: {e}")
    return out if len(out) >= 1 else None


def evaluate_extractions_openrouter(
    model_responses: List[ModelResponse],
    model: Optional[str] = None,
) -> ExpertEvaluation:
    """
    Run the same expert prompt as :meth:`ExpertEvaluator.evaluate_extractions` but via OpenRouter
    (e.g. GPT-5.4 class models). Requires ``OPENROUTER_API_KEY`` and optional ``OPENROUTER_EXPERT_MODEL``.
    """
    from openai import OpenAI

    api_key = config.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    model_id = model or config.OPENROUTER_EXPERT_MODEL
    headers = {}
    if getattr(config, "OPENROUTER_HTTP_REFERER", ""):
        headers["HTTP-Referer"] = config.OPENROUTER_HTTP_REFERER
    headers["X-Title"] = getattr(config, "OPENROUTER_APP_NAME", "WikistimProject")

    client = OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers=headers,
    )

    successful = [r for r in model_responses if r.extraction is not None]
    if not successful:
        raise ValueError("No successful extractions to evaluate")

    if len(successful) == 1:
        body = ExpertEvaluator._body_from_extraction(successful[0].extraction)
        return ExpertEvaluation(
            best_extraction=body,
            reasoning="Only one model output loaded.",
            field_level_decisions={},
            agreement_score=None,
        )

    payloads = [
        {
            "provider": r.provider,
            "model_name": r.model_name,
            "extraction": r.extraction.model_dump(mode="json"),
        }
        for r in successful
    ]
    responses_json = json.dumps(payloads, indent=2, ensure_ascii=False)
    raw_prompt = load_prompt(config.OPENAI_EXPERT_PROMPT)
    prompt = Template(raw_prompt).safe_substitute(responses_json=responses_json)

    system_prompt_path = config.OPENAI_EXPERT_SYSTEM_PROMPT
    system_content = (
        load_prompt(system_prompt_path)
        if system_prompt_path and system_prompt_path.exists()
        else "You are an expert at evaluating and comparing scientific paper extractions."
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        timeout=180.0,
    )
    response_text = (response.choices[0].message.content or "").strip()
    data = _extract_json_object_from_llm_text(response_text)
    return ExpertEvaluator._parse_expert_judgment_response(data)


if __name__ == "__main__":

    default_pdf = config.TRAINING_DIR / "pdf_17.pdf"
    if not default_pdf.exists():
        pdfs = sorted(config.TRAINING_DIR.glob("*.pdf"))
        default_pdf = pdfs[0] if pdfs else config.TRAINING_DIR / "pdf_17.pdf"

    pdf_path = default_pdf
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])

    evaluator = ExpertEvaluator()

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    stem = pdf_path.stem
    candidates = sorted(
        (
            list(output_dir.glob(f"{stem}_extraction_*.json"))
            + list(output_dir.glob(f"{stem}_extraction.json"))
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    model_responses: Optional[List[ModelResponse]] = None
    loaded_from_output = False

    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "model_responses" in payload:
                model_responses = [
                    ModelResponse.model_validate(mr) for mr in payload["model_responses"]
                ]
                loaded_from_output = True
                print("\n=== Loaded model responses from output ===")
                print(f"File: {candidate}")
                break
        except Exception:
            continue

    if not loaded_from_output:
        model_responses = load_model_responses_from_split_dir(output_dir, stem)
        if model_responses:
            loaded_from_output = True
            print(f"\n=== Loaded per-model JSON from {output_dir} ({stem}_*.json) ===")

    if not loaded_from_output:
        print(f"\n=== Running extractors on {pdf_path.name} (no saved outputs found) ===")
        model_responses = evaluator.run_extractors(pdf_path)

    for r in model_responses:
        status = "OK" if r.extraction else f"FAILED: {r.error}"
        print(f"  {r.provider}: {status}")

    # 2 Expert Evaluate
    print("\n=== Expert evaluation ===")
    evaluation = evaluator.evaluate_extractions(model_responses)