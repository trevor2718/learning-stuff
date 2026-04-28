"""
Feature extraction via OpenRouter (OpenAI-compatible API).

Mirrors :class:`OpenAIExtractor` behavior but routes requests to OpenRouter
so you can use models such as ``openai/gpt-4o-mini`` without calling the
direct OpenAI API (helps with separate rate limits / billing).

PDFs are sent as **extracted text** (see ``utils.pdf_utils.extract_pdf_text``),
because OpenRouter does not support OpenAI's file upload + Responses API.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from openai.types.file_object import FileObject

# ---- local run ----
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from models.pubmed import PubMedAPI
from models.schemas import ModelResponse, ResearchPaperExtraction, TestPaperExtraction
from utils.logging_config import logger
from utils.pdf_utils import extract_pdf_text, validate_pdf_file
from utils.prompts_utils import load_prompt


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object in model response")
    return json.loads(text[start:end])


class OpenRouterExtractor:
    """GPT-class extraction through OpenRouter's OpenAI-compatible endpoint."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or config.OPENROUTER_API_KEY
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment or config")

        headers = {}
        if getattr(config, "OPENROUTER_HTTP_REFERER", ""):
            headers["HTTP-Referer"] = config.OPENROUTER_HTTP_REFERER
        title = getattr(config, "OPENROUTER_APP_NAME", None) or "WikistimProject"
        headers["X-Title"] = title

        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=key,
            default_headers=headers,
        )
        self.model = config.OPENROUTER_MODEL
        self.mini_model = config.OPENROUTER_MINI_MODEL
        self.name = "OpenRouterExtractor"

        logger.info(
            f"OpenRouter extractor initialized (model={self.model}, mini={self.mini_model})"
        )

    def _chat_json(self, system: str, user: str, model: str, timeout: float = 180.0) -> dict:
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=config.DEFAULT_TEMPERATURE,
            timeout=timeout,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Empty response from OpenRouter")
        return _extract_json_object(content)

    def extract_essential_features(self, file_path: Path) -> TestPaperExtraction:
        file_path = validate_pdf_file(file_path)
        text = extract_pdf_text(file_path, max_pages=1)
        prompt = load_prompt(config.OPENAI_ESSENTIAL_FEATURES_PROMPT)
        user = f"{prompt}\n\n--- DOCUMENT TEXT (first page) ---\n\n{text}"
        system = (
            "You extract bibliographic metadata from academic paper text. "
            "Reply with a single JSON object only, matching the schema requested by the user."
        )
        data = self._chat_json(system, user, self.mini_model, timeout=90.0)
        return TestPaperExtraction.model_validate(data)

    def extract_research_features(self, file_path: Path) -> ResearchPaperExtraction:
        logger.info("Extracting all features with OpenAI")
        
        text = extract_pdf_text(file_path)
        
        prompt = load_prompt(config.OPENAI_FEATURES_PROMPT)
        
        response = self.client.responses.parse(
            model=self.model,
            # temperature=self.temperature,
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "low",
            }],
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{prompt}\n \n --- DOCUMENT TEXT ---\n {text}",
                        }
                    ]
                }
            ],
            text_format=ResearchPaperExtraction,
        )
        
        logger.info("All features extracted with OpenAI")
        data = response.output_parsed
        return data

    def extract_features(self, file_path: Path) -> ModelResponse:
        start = time.time()
        fp = Path(file_path)
        try:
            extraction = self.extract_research_features(fp)
            pubmed_api = PubMedAPI()
            pmid = pubmed_api.find_pmid_by_metadata(
                extraction.authors,
                extraction.title,
                extraction.year_of_publication,
                extraction.doi,
            )
            pubmed_link = pubmed_api.get_pubmed_link_by_pmid(pmid) if pmid else None
            extraction.pubmed_hyperlink = pubmed_link if pubmed_link else None
            if extraction.field_confidence:
                extraction.field_confidence.pubmed_hyperlink = 1.0 if pubmed_link else 0.0

            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{fp.stem}_openai_openrouterapi.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(extraction.model_dump(), f, indent=2, default=str)
            logger.info(f"OpenAI OpenRouter API output saved to {output_path}")

            return ModelResponse(
                model_name=self.model,
                provider="OpenAI",
                extraction=extraction,
                processing_time=time.time() - start,
                error=None,
            )
        except Exception as e:
            logger.error(f"OpenRouter extraction failed: {e}")
            return ModelResponse(
                model_name=self.model,
                provider="OpenAI",
                extraction=None,
                processing_time=time.time() - start,
                error=str(e),
            )


if __name__ == "__main__":
    ex = OpenRouterExtractor()
    # ex.extract_features(config.TRAINING_DIR / "pdf_0.pdf")
    for num in range (0, 23):
        try:
            ex.extract_features(config.TRAINING_DIR / f"pdf_{num}.pdf")
        except Exception as e:
            logger.error(f"Error extracting features for pdf_{num}: {e}")
