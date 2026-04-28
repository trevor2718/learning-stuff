"""
Google Gemini-based feature extraction.
Each document page = 258 tokens
"""

import time
import base64
from pathlib import Path
from typing import Optional
import json

from google import genai
from google.genai import types as genai_types

# ---- remove. used for direct testing of API 
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# ----

from config import config
from pydantic import ValidationError

from models.schemas import (
    TestPaperExtraction, 
    ResearchPaperExtraction, 
    TestPaperConfidence,
    ResearchPaperConfidence,
    ModelResponse,
)

from models.pubmed import PubMedAPI
from utils.logging_config import logger
from utils.prompts_utils import load_prompt, load_unfilled_features_prompt
from utils.pdf_utils import read_pdf_as_base64, get_pages


class GeminiExtractor:
    """Handles feature extraction using Google's Gemini models."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.client = genai.Client(api_key=self.api_key)
        self.model = config.GOOGLE_MODEL
        self.temperature = config.DEFAULT_TEMPERATURE
        self.name = "GeminiExtractor"
        
        logger.info("Gemini Extractor initialized.")    
    
    def _test_connection(self) -> dict:
        """
        Test the API connection 
        
        Returns:
            dict: Status information about the API connection
        """
        try:
            # Make a minimal test API call
            response = self.client.models.generate_content(
                model=self.model,
                contents="Hi",
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=10
                )
            )
            
            logger.info(f"Gemini connection success. model: {self.model}")
            
            return {
                'valid': True,
                'model': self.model,
                'message': 'Success',
                'response_text': response.text[:50] if response.text else None
            }
        
        except Exception as e:
            error_str = str(e)
            
            # Parse common errors
            if "API_KEY_INVALID" in error_str or "invalid API key" in error_str:
                message = "Invalid GOOGLE_API_KEY"
            elif "429" in error_str or "quota" in error_str.lower():
                message = "Rate limit or quota exceeded"
            elif "403" in error_str:
                message = "API key doesn't have permission. Enable Gemini API in Google Cloud Console."
            else:
                message = f"Connection failed: {error_str}"
            
            logger.error(f"{message}")
            
            return {
                'valid': False,
                'model': self.model,
                'message': message,
                'response_text' : error_str,
            }
    
    def _upload_file(self, file_path: Path):
        logger.debug(f"Uploading file to Gemini: {file_path.name}")
        uploaded = self.client.files.upload(file=str(file_path))
        logger.debug(f"File uploaded: {uploaded.name}")
        return uploaded
 
    def _delete_file(self, uploaded_file) -> None:
        """Delete an uploaded file to avoid accumulating storage."""
        try:
            self.client.files.delete(name=uploaded_file.name)
            logger.debug(f"Deleted Gemini file: {uploaded_file.name}")
        except Exception as e:
            logger.warning(f"Could not delete Gemini file {uploaded_file.name}: {e}")
            
    def extract_essential_features(self, file_path: Path) -> TestPaperExtraction:
        """
        Extract essential features (title, authors) from PDF.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            EssentialFeatures object
        """
        
        logger.debug("Extracting essential features with Gemini")
        
        shortened_pdf = get_pages(file_path, page_numbers=[0])
        uploaded_file = self._upload_file(shortened_pdf)
 
        prompt = load_prompt(config.GOOGLE_ESSENTIAL_FEATURES_PROMPT)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded_file, prompt],
            config=genai_types.GenerateContentConfig(
                temperature=self.temperature,
                # system_instruction=,
                max_output_tokens=1024
            )
        )
        
        # Parse response
        response_text = response.text.replace('```json', '').replace('```', '').strip()

        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        # EXtract JSON 
        try:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            return TestPaperExtraction(**data)

        except json.JSONDecodeError:
            # JSON parsing failed
            logger.error(f"Invalid Gemini JSON: {response_text}")
            raise ValueError("LLM didn't return valid JSON")

        except ValidationError as e:
            # Pydantic validation failed
            logger.error(f"Invalid Gemini data structure: {e}")
            raise ValueError("LLM returned wrong fields")
        
        finally:
            self._delete_file(uploaded_file)
        
    def extract_research_features(self, file_path: Path) -> ResearchPaperExtraction:
        """
        Extract all features from PDF.
        
        Args:
            file_path: Path to the PDF file
            essential_features: Previously extracted features (optional)
        
        Returns:
            Complete ResearchPaperExtraction
        """
        logger.info("Extracting all features with Gemini")
    
        uploaded_file = self._upload_file(file_path)
        
        prompt = load_prompt(config.GOOGLE_FEATURES_PROMPT)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded_file, prompt],
            config=genai_types.GenerateContentConfig(
                temperature=self.temperature,
                # system_instruction=,
                max_output_tokens=4096
            )
        )
        
        # Parse response
        response_text = response.text.replace('```json', '').replace('```', '').strip()

        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        try:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            return ResearchPaperExtraction(**data)

        except json.JSONDecodeError:
            # JSON parsing failed
            logger.error(f"Invalid Gemini JSON: {response_text}")
            raise ValueError("LLM didn't return valid JSON")

        except ValidationError as e:
            # Pydantic validation failed
            logger.error(f"Invalid Gemini data structure: {e}")
            raise ValueError("LLM returned wrong fields")
        finally:
            self._delete_file(uploaded_file)
    
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
 
            # Resolve PubMed link
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
 
            processing_time = time.time() - start_time
 
            # Save output to output/gemini/<pdf_stem>_gemini.json
            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{file_path.stem}_gemini.json"
            with open(output_path, "w") as f:
                json.dump(extraction.model_dump(), f, indent=2, default=str)
            logger.info(f"Gemini output saved to {output_path}")
 
            return ModelResponse(
                model_name=self.model,
                provider="Google",
                extraction=extraction,
                processing_time=processing_time,
                error=None,
            )

        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Gemini extraction failed: {str(e)}")
            
            return ModelResponse(
                model_name=self.model,
                provider="Google",
                extraction=None,
                processing_time=processing_time,
                error=str(e)
            )


# --- Debugging
if __name__ == '__main__':
    num = 0
    pdf_test = config.TRAINING_DIR / f"pdf_{num}.pdf"
    geminiextractor = GeminiExtractor()
    output = geminiextractor.extract_features(pdf_test)
    print(output)
