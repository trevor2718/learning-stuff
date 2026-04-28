"""
Main extraction orchestrator that coordinates multiple models and expert evaluation.
"""

import time
import json
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.schemas import (
    ExpertEvaluation,
    ExtractionResult,
    ModelResponse,
    ResearchPaperExtraction,
    SimilarityResult,
    TestPaperExtraction,
)

from models.openai_extractor import OpenAIExtractor
from models.claude_extractor import ClaudeExtractor
from models.gemini_extractor import GeminiExtractor
from models.expert_evaluator import ExpertEvaluator

from utils.logging_config import logger, log_model_response

from utils.pdf_utils import validate_pdf_file, get_file_size_mb


class FeatureExtractor:
    """
    Main orchestrator for extracting features from research papers.
    Coordinates multiple models and expert evaluation.
    """
    
    def __init__(
        self,
        use_openai: bool = True,
        use_gemini: bool = True,
        use_claude: bool = True,
        use_expert_evaluation: bool = True,
        use_similarity_scoring: bool = True,
        ):
        """
        Initialize the feature extractor.
        
        Args:
            use_openai: Whether to use direct OpenAI API (disabled if use_openrouter)
            use_openrouter: Use OpenRouter for the GPT extraction slot instead of OpenAI
            use_claude: Whether to use Claude model
            use_expert_evaluation: Whether to use expert evaluation
            use_similarity_scoring: After expert consensus, score vs ground_truth/ if present
        """
        self.use_openai = use_openai and not use_openrouter
        self.use_openrouter = use_openrouter
        self.use_gemini = use_gemini
        self.use_claude = use_claude
        self.use_expert_evaluation = use_expert_evaluation
        self.use_similarity_scoring = use_similarity_scoring
        self._similarity_scorer = None
        
        # Initialize extractors
        self.openrouter_extractor = OpenRouterExtractor() if use_openrouter else None
        self.openai_extractor = OpenAIExtractor() if self.use_openai else None
        self.claude_extractor = ClaudeExtractor() if use_claude else None
        self.gemini_extractor = GeminiExtractor() if use_gemini else None
        self.expert_evaluator = ExpertEvaluator() if use_expert_evaluation else None
        self.parallel = False

        logger.info(f"Extraction Params:")
        logger.info(f"  - OpenRouter (replaces OpenAI): {use_openrouter}")
        logger.info(f"  - OpenAI (direct): {self.use_openai}")
        logger.info(f"  - Gemini: {use_gemini}")
        logger.info(f"  - Claude: {use_claude}")
        logger.info(f"  - Expert Evaluation: {use_expert_evaluation}")
        logger.info(f"  - Ground-truth similarity: {use_similarity_scoring}")
    
    def extract_from_pdf(self, pdf_path: str | Path) -> ExtractionResult:
        """
        Extract features from a single PDF using all configured models.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            ExtractionResult with all model responses and expert evaluation
        """
        pdf_path = validate_pdf_file(pdf_path)
        
        start_time = time.time()
        
        # Essential Feature extraction
        # essential_features = self._extract_essential_features(pdf_path)
        
        # if essential_features:
        #     logger.info(f"Title: {essential_features.title}")
        #     logger.info(f"Authors: {essential_features.authors}")
        
        # Feature Extraction
        model_responses = self._run_all_models(pdf_path)
        
        # Log results
        for response in model_responses:
            log_model_response(
                logger,
                response.model_name,
                response.provider,
                (response.error is None),
                response.processing_time,
                response.error
            )
        
        # Expert evaluation
        expert_evaluation = None
        if self.use_expert_evaluation and len([r for r in model_responses if r.extraction]) > 0:
            try:
                logger.info("Running expert evaluation...")
                expert_evaluation = self.expert_evaluator.evaluate_extractions(model_responses)
                logger.info(
                    f"Agreement score: {expert_evaluation.agreement_score}"
                    if expert_evaluation.agreement_score is not None
                    else "Agreement score: n/a"
                )
            except Exception as e:
                logger.error(f"Expert evaluation failed: {str(e)}")

        ground_truth_similarity = None
        if (
            self.use_similarity_scoring
            and expert_evaluation is not None
        ):
            ground_truth_similarity = self._score_expert_vs_ground_truth(
                pdf_path, expert_evaluation
            )

        total_time = time.time() - start_time
        
        result = ExtractionResult(
            paper_path=str(pdf_path),
            essential_features=None,
            model_responses=model_responses,
            expert_evaluation=expert_evaluation,
            ground_truth_similarity=ground_truth_similarity,
            total_processing_time=total_time
        )
        
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        return result
    
    def _get_similarity_scorer(self):
        """Lazy-load sentence-transformers model"""
        if self._similarity_scorer is None:
            from models.similarity_scorer import SimilarityScorer

            self._similarity_scorer = SimilarityScorer()
        return self._similarity_scorer

    def _score_expert_vs_ground_truth(
        self, pdf_path: Path, expert_evaluation: ExpertEvaluation
    ) -> Optional[SimilarityResult]:
        """
        Compare expert best_extraction to WikiStim JSON in ground_truth/ using SimilarityScorer.
        """
        from models.similarity_scorer import load_ground_truth

        try:
            gt = load_ground_truth(pdf_path)
            if gt is None:
                logger.info("No ground truth file for this PDF; skipping similarity scoring.")
                return None

            consensus = ResearchPaperExtraction.model_validate(
                expert_evaluation.best_extraction.model_dump(mode="json")
            )
            logger.info("Running similarity scoring against ground truth...")
            return self._get_similarity_scorer().score(
                consensus, gt, paper_path=str(pdf_path)
            )
        except Exception as e:
            logger.warning(f"Ground-truth similarity scoring skipped: {e}")
            return None

    def _extract_essential_features(self, pdf_path: Path) -> Optional[TestPaperExtraction]:
        """Extract basic features needed for author and title ID with OpenAI Mini Model."""
        if self.openrouter_extractor:
            try:
                logger.debug("Extracting essential features (OpenRouter)...")
                return self.openrouter_extractor.extract_essential_features(pdf_path)
            except Exception as e:
                logger.error(f"Failed to extract essential features: {str(e)}")
                return None
        if not self.openai_extractor:
            return None
        
        try:
            logger.debug("Extracting essential features...")
            return self.openai_extractor.extract_essential_features(pdf_path)
        except Exception as e:
            logger.error(f"Failed to extract essential features: {str(e)}")
            return None
    
    def _run_all_models(self, pdf_path: Path) -> List[ModelResponse]:
        """
        Run all configured models on the PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of ModelResponse objects
        """
        responses = []
        
        if self.parallel:
            responses = self._run_models_parallel(pdf_path)
        else:
            responses = self._run_models_sequential(pdf_path)
        
        return responses
    
    def _run_models_sequential(self, pdf_path: Path) -> List[ModelResponse]:
        """Run models one after another."""
        responses = []
        
        if self.openrouter_extractor:
            logger.info("Running OpenRouter extraction...")
            response = self.openrouter_extractor.extract_features(pdf_path)
            responses.append(response)
        elif self.openai_extractor:
            logger.info("Running OpenAI extraction...")
            response = self.openai_extractor.extract_features(pdf_path)
            responses.append(response)
        
        if self.claude_extractor:
            logger.info("Running Claude extraction...")
            response = self.claude_extractor.extract_features(pdf_path)
            responses.append(response)
        
        if self.gemini_extractor:
            logger.info("Running Gemini extraction...")
            response = self.gemini_extractor.extract_features(pdf_path)
            responses.append(response)
            
        return responses
    
    def _run_models_parallel(self, pdf_path: Path) -> Optional[ModelResponse]:
        """//TODO: if a valuable feature"""
        return None
    
    def save_result(
        self,
        result: ExtractionResult,
        output_path: Optional[Path] = None
        ) -> Path:
        """
        Save extraction result to JSON file.
        
        Args:
            result: ExtractionResult to save
            output_path: Path to save file. If None, auto-generates name.
        
        Returns:
            Path to saved file
        """
        pdf_stem = Path(result.paper_path).stem

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"output/{pdf_stem}_extraction_{timestamp}.json")
        else:
            output_path = Path(output_path)
            if output_path.exists() and output_path.is_dir():
                output_path = output_path / f"{pdf_stem}_extraction.json"
            elif output_path.suffix == "" and not output_path.exists():
                # Treat bare path like "output" as a directory
                output_path = output_path / f"{pdf_stem}_extraction.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = result.model_dump(mode="json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        if result.expert_evaluation is not None:
            expert_path = output_path.parent / f"{pdf_stem}_expert.json"
            expert_payload = result.expert_evaluation.model_dump(mode="json")
            with open(expert_path, "w", encoding="utf-8") as f:
                json.dump(expert_payload, f, indent=2)
            logger.info(f"Expert evaluation saved to: {expert_path}")

            consensus_path = output_path.parent / f"{pdf_stem}_consensus.json"
            consensus = result.expert_evaluation.best_extraction.model_dump(mode="json")
            with open(consensus_path, "w", encoding="utf-8") as f:
                json.dump(consensus, f, indent=2)
            logger.info(f"Consensus saved to: {consensus_path}")

        if result.ground_truth_similarity is not None:
            sim_path = output_path.parent / f"{pdf_stem}_similarity.json"
            with open(sim_path, "w", encoding="utf-8") as f:
                json.dump(
                    result.ground_truth_similarity.model_dump(mode="json"),
                    f,
                    indent=2,
                )
            logger.info(f"Ground-truth similarity saved to: {sim_path}")

        logger.info(f"Results saved to: {output_path}")
        return output_path
