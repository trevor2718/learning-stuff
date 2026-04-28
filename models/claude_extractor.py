"""
Anthropic Claude-based feature extraction.


"""

import time
import base64
import json
from pathlib import Path
from typing import Optional

# --------------- for running as main for debugging
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# ---------------

import anthropic
from anthropic import Anthropic


from config import config
from models.schemas import (
    TestPaperExtraction, 
    ResearchPaperExtraction, 
    TestPaperConfidence,
    ResearchPaperConfidence,
    ModelResponse,
)
    
from utils.logging_config import logger
from utils.prompts_utils import load_prompt, load_unfilled_features_prompt
from utils.pdf_utils import get_pages

from models.pubmed import PubMedAPI


class ClaudeExtractor:
    """Handles feature extraction using Anthropic's Claude models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Claude extractor.
        
        Args:
            api_key: Anthropic API key. If None, uses config default.
        """
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.client = Anthropic(api_key=self.api_key)
        self.model = config.ANTHROPIC_MODEL
        self.temperature = config.DEFAULT_TEMPERATURE
        self.name = "ClaudeExtractor"
        
        logger.info("Claude extractor initialized.")
    
    def _test_connection(self) -> bool:
        """
        Test the API connection and return diagnostic info.
        """
        try:
            # minimal test API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": "Hi"
                    }
                ]
            )
            
            logger.info(f"Claude API connection successful. Using model: {self.model}")
            print(response)
            
            return True
        
        except Exception as e:
            error_str = str(e)
            
            # Parse common errors
            if "401" in error_str or "invalid x-api-key" in error_str:
                message = "Invalid ANTHROPIC_API_KEY"
            elif "429" in error_str or "rate_limit" in error_str:
                message = "Rate limit exceeded."
            elif "overloaded" in error_str:
                message = "Anthropic API is overloaded."
            elif "insufficient_quota" in error_str or "credit" in error_str:
                message = "Credits expired."
            else:
                message = f"Connection failed: {error_str}"
            
            logger.error(f"{message}")
            
            return False
    
    def extract_essential_features(self, file_path: Path) -> TestPaperExtraction:
        """
        Extract essential features (title, authors) from PDF.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            TestPaperExtraction object
        """

        logger.debug("Extracting essential features with Claude")
        
        # use first page to make a low token call for pubmed link or fallback on title/authors for lookup
        shortened_pdf = get_pages(file_path, page_numbers=[0])
        
        # Anthropic allows for base64 encoded data.
        with open(shortened_pdf, "rb") as f:
            pdf_data = f.read()
        pdf_base64 = base64.standard_b64encode(pdf_data).decode("utf-8")
        
        prompt = load_prompt(config.ANTHROPIC_ESSENTIAL_FEATURES_PROMPT)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        
        # Parse the response
        response_text = response.content[0].text
        
        logger.debug(response_text)
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]

        data = json.loads(json_str)
        return TestPaperExtraction(**data)
    
    def extract_research_features(self, file_path: Path) -> ResearchPaperExtraction:
        """
        Extract all features from PDF.
        
        Args:
            file_path: Path to the PDF file
            essential_features: Previously extracted features (optional)
        
        Returns:
            Complete ResearchPaperExtraction
        """
        logger.debug("Extracting all features with Claude")
        
        prompt = load_prompt(config.ANTHROPIC_RESEARCH_FEATURES_PROMPT)
        
        with open(file_path, "rb") as f:
            pdf_data = f.read()
        pdf_base64 = base64.standard_b64encode(pdf_data).decode("utf-8")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
        
        # Parse the response
        response_text = response.content[0].text
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        data = json.loads(json_str)
        return ResearchPaperExtraction(**data) # data validation of output, note format impacted by prompt design
    
    def extract_features(self, file_path: Path) -> ModelResponse:
        """
        Complete feature extraction pipeline for a PDF.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            ModelResponse containing extraction results
        """
        start_time = time.time()
        
        try:        
            # Extract all features
            extraction = self.extract_research_features(file_path)
            
            # Get Pubmed link
            pubmed_api = PubMedAPI()
            pmid = pubmed_api.find_pmid_by_metadata(extraction.authors, extraction.title, extraction.year_of_publication, extraction.doi)
            pubmed_link = pubmed_api.get_pubmed_link_by_pmid(pmid) if pmid else None
            # print(pubmed_link)
            
            extraction.pubmed_hyperlink = pubmed_link if pubmed_link else None
            if extraction.field_confidence:
                extraction.field_confidence.pubmed_hyperlink = 1.0 if pubmed_link else 0.0
            
            processing_time = time.time() - start_time
            
            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{file_path.stem}_claude.json"
            with open(output_path, "w") as f:
                json.dump(extraction.model_dump(), f, indent=2, default=str)
            logger.info(f"Claude output saved to {output_path}")
            
            return ModelResponse(
                model_name=self.model,
                provider="Anthropic",
                extraction=extraction,
                processing_time=processing_time,
                error=None
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Claude extraction failed: {str(e)}")
        
            
            return ModelResponse(
                model_name=self.model,
                provider="Anthropic",
                extraction=None,
                processing_time=processing_time,
                error=str(e)
            )

    
if __name__ == "__main__":
    
    for num in range(0, 23):
        pdf_training = config.TRAINING_DIR / f"pdf_{num}.pdf"
        claude_extractor = ClaudeExtractor()
        output = claude_extractor.extract_features(pdf_training)
        print(output)
    
    
