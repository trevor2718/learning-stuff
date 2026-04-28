"""
OpenAI feature extraction
"""

import time
import json
from pathlib import Path
from typing import Optional, Tuple

# ---- remove. used for direct testing of API 
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# ----

from openai import OpenAI
from openai.types.file_object import FileObject

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
from utils.pdf_utils import get_pages

class OpenAIExtractor:
    """Handles feature extraction using OpenAI's GPT models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI extractor.
        
        Args:
            (optional) api_key: OpenAI API key else config api key
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.OPENAI_MODEL
        self.mini_model = config.OPENAI_MINI_MODEL
        self.temperature = config.DEFAULT_TEMPERATURE
        
        self.name = "OpenAIExtractor"
        
        logger.info("OpenAI Extractor initialized.")
    
    def _test_connection(self) -> bool:
        """
        Validate the API key by making a test API call.
        
        Returns:
            bool: True if API key is valid, False otherwise
        
        Raises:
            Exception: If API call fails with details
        """
        try:
            # Make a minimal API call to test the key
            response = self.client.models.list()
            logger.info("OpenAI API key validated")
            return True
        
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Incorrect API key" in error_msg:
                logger.error("Invalid OpenAI API key")
                raise ValueError(f"Invalid OpenAI API key: {error_msg}")
            elif "429" in error_msg:
                logger.warning("OpenAI API rate limit reached")
                raise ValueError(f"Rate limit reached: {error_msg}")
            else:
                logger.error(f"OpenAI API error: {error_msg}")
                raise Exception(f"OpenAI API error: {error_msg}")
        
    def _upload_file(self, file_path: Path) -> FileObject:
        """
        Upload a PDF file to OpenAI.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            FileObject containing file ID and metadata
        """
        logger.debug(f"Uploading file to OpenAI: {file_path}")
        
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(
                file=f,
                purpose="user_data"
            )
        
        logger.debug(f"File uploaded successfully: {file_obj.id}")
        
        return file_obj
    
    def _delete_file(self, file_id: str) -> None:
        """Delete an uploaded file to avoid accumulating storage."""
        try:
            self.client.files.delete(file_id)
            logger.debug(f"Deleted OpenAI file: {file_id}")
        except Exception as e:
            logger.warning(f"Could not delete OpenAI file {file_id}: {e}")

    def extract_essential_features(self,file_path: Path) -> TestPaperExtraction:
        """
        Extract essential features (title, authors) from PDF.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            EssentialFeatures
        """
        logger.info("Extracting essential features with OpenAI mini model")
        
        shortened_pdf = get_pages(file_path, page_numbers=[0])
        
        file_obj = self._upload_file(shortened_pdf)
        
        try:
            response = self.client.responses.parse(
                model=self.mini_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": file_obj.id, 
                            },
                            {
                                "type": "input_text",
                                "text": load_prompt(config.OPENAI_ESSENTIAL_FEATURES_PROMPT),
                            },
                        ]
                    }
                ],
                text_format=TestPaperExtraction
            )
        
            data = response.output_parsed
            logger.info("Essential features extracted with OpenAI")
            
            return data
        
        finally:
            self._delete_file(file_obj.id)
    
    def extract_research_features(self, file_path : Path) -> ResearchPaperExtraction:
        """
        Extract all features from PDF using web search for metadata.
        
        Args:
            essential_features: Previously extracted title and authors
            file_obj: Uploaded file object
        
        Returns:
            ResearchPaperExtraction
        """
        
        logger.info("Extracting all features with OpenAI")
        
        file_obj = self._upload_file(file_path)
        
        try:
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
                                "type": "input_file",
                                "file_id": file_obj.id,
                            },
                            {
                                "type": "input_text",
                                "text": prompt,
                            }
                        ]
                    }
                ],
                text_format=ResearchPaperExtraction,
            )
            
            logger.info("All features extracted with OpenAI")
            data = response.output_parsed
            return data
        
        finally:
            self._delete_file(file_obj.id)
    
    def extract_features(self, file_path: Path) -> ModelResponse:
        """
        Complete feature extraction pipeline for a PDF.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            ModelResponse containing extraction results
        """
        start_time = time.time() #time logger
        
        try:
            extraction = self.extract_research_features(file_path)
            logger.info("OpenAI extraction success.")
            
            # Get Pubmed link
            pubmed_api = PubMedAPI()
            pmid = pubmed_api.find_pmid_by_metadata(extraction.authors, extraction.title, extraction.year_of_publication, extraction.doi)
            pubmed_link = pubmed_api.get_pubmed_link_by_pmid(pmid) if pmid else None
            # print(pubmed_link)
            
            extraction.pubmed_hyperlink = pubmed_link if pubmed_link else None
            if extraction.field_confidence:
                extraction.field_confidence.pubmed_hyperlink = 1.0 if pubmed_link else 0.0
            
            # Output Logger
            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{file_path.stem}_openai.json"
            with open(output_path, "w") as f:
                json.dump(extraction.model_dump(), f, indent=2, default=str)
            logger.info(f"OpenAI output saved to {output_path}")
            
            processing_time = time.time() - start_time #time logger
            
            return ModelResponse(
                model_name=self.model,
                provider="OpenAI",
                extraction=extraction,
                processing_time=processing_time,
                error=None
            )
        
        except Exception as e:
            processing_time = time.time() - start_time #time logger
            logger.error(f"OpenAI extraction failed: {str(e)}")
            
            # Error information.            
            return ModelResponse(
                model_name=self.model,
                provider="OpenAI",
                extraction=None,
                processing_time=processing_time,
                error=str(e)
            )
            
            
if __name__ == '__main__':
    openaiextractor = OpenAIExtractor()
    openaiextractor._test_connection()
    # essential_features = openaiextractor.extract_essential_features(config.TRAINING_DIR / "pdf_1.pdf")
    # print(essential_features)
    output = openaiextractor.extract_features(config.TRAINING_DIR / "pdf_1.pdf")
    print(output)
    
