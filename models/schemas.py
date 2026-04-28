"""
Feature Extraction.

Pydantic models for structured data extraction from research papers.
Defines the schema for extracted features.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


#
# Enums
#
class StudyDesign(str, Enum):
    CLINICAL_TRIAL = "clinical_trial"
    COST_ANALYSIS = "cost_analysis"
    LABORATORY_STUDY = "laboratory_study"
    COMPUTER_MODELING = "computer_modeling_study"
    SCIATIC_NERVE_LIGATURE = "sciatic_nerve_ligature_modeling_study"
    CASE_STUDY = "case_study_uncontrolled"
    CASE_SERIES = "case_series_uncontrolled"
    CASE_CONTROL = "case_control_study"
    PROSPECTIVE_COHORT = "prospective_cohort_controlled"
    RETROSPECTIVE_COHORT = "retrospective_cohort_controlled"
#
# Confidence (0-1 float)
#
class TestPaperConfidence(BaseModel):
    """Per-field confidence scores for TestPaperExtraction (0.0 – 1.0)."""
    title: float = Field(default=0.0, ge=0.0, le=1.0)
    authors: float = Field(default=0.0, ge=0.0, le=1.0)
    year_of_publication: float = Field(default=0.0, ge=0.0, le=1.0)
    doi: float = Field(default=0.0, ge=0.0, le=1.0)
    
class ResearchPaperConfidence(BaseModel):
    """Per-field confidence scores for ResearchPaperExtraction (0.0 – 1.0)."""
    authors: float = Field(default=0.0, ge=0.0, le=1.0)
    title: float = Field(default=0.0, ge=0.0, le=1.0)
    journal_name: float = Field(default=0.0, ge=0.0, le=1.0)
    volume_issue_pages: float = Field(default=0.0, ge=0.0, le=1.0)
    year_of_publication: float = Field(default=0.0, ge=0.0, le=1.0)
    pubmed_hyperlink: float = Field(default=0.0, ge=0.0, le=1.0)
    study_design: float = Field(default=0.0, ge=0.0, le=1.0)
    study_question: float = Field(default=0.0, ge=0.0, le=1.0)
    population_assessed: float = Field(default=0.0, ge=0.0, le=1.0)
    follow_up_duration: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome_measures: float = Field(default=0.0, ge=0.0, le=1.0)
    inclusion_criteria: float = Field(default=0.0, ge=0.0, le=1.0)
    number_of_participants_in_study: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_to_study_question: float = Field(default=0.0, ge=0.0, le=1.0)
 
    def mean(self) -> float:
        """Mean confidence of all fields"""
        values = list(self.model_dump().values())
        return sum(values) / len(values) if values else 0.0

    

#
# Data model
#

class TestPaperExtraction(BaseModel):
    """Eseential features needed for web search."""
    
    title: str = Field(description="Title of the research paper")
    authors: str = Field(description="Authors in format: Last name Initials, separated by commas")
    year_of_publication : int = Field(description="Year of publication")
    doi : Optional[str] = Field(default=None, description="DOI of paper if available")
    
    field_confidence: Optional[TestPaperConfidence] = Field(
        default=None,
        description="Per-field confidence scores assigned by the model"
    )
 
    @property
    def mean_confidence(self) -> Optional[float]:
        return self.field_confidence.mean() if self.field_confidence else None


class ResearchPaperExtractionBody(BaseModel):
    """
    Research paper feature fields only (no per-model confidence).
    Used for expert-chosen output and human-truth comparison input.
    """

    # Publication metadata
    authors: str = Field(description="List of authors")
    title: str = Field(description="Title of the paper")
    journal_name: str = Field(description="Name of the journal")

    volume_issue_pages: str = Field(
        description="Volume, issue, and page numbers (e.g., '9(1):11441')"
    )

    year_of_publication: int = Field(description="Year of publication")

    doi: Optional[str] = Field(default=None, description="DOI of the paper if available")

    pubmed_hyperlink: Optional[str] = Field(
        default=None,
        description="PubMed link to the article",
    )

    # Study characteristics
    study_design: StudyDesign = Field(description="Study design")
    study_question: str = Field(description="Main research question")
    population_assessed: str = Field(description="Population/sample studied")
    follow_up_duration: str = Field(description="Duration of follow-up period")
    outcome_measures: str = Field(description="Primary outcome measures")
    inclusion_criteria: str = Field(description="Study inclusion criteria")
    number_of_participants_in_study: Optional[int] = Field(
        description="Number or range of participants"
    )
    answer_to_study_question: str = Field(
        description="Main findings/answer to research question"
    )


class ResearchPaperExtraction(ResearchPaperExtractionBody):
    """Extraction schema for research paper features (includes optional confidence)."""

    field_confidence: Optional[ResearchPaperConfidence] = Field(
        default=None,
        description="Per-field confidence scores assigned by the model",
    )

    @property
    def mean_confidence(self) -> Optional[float]:
        return self.field_confidence.mean() if self.field_confidence else None

#
# Pipeline Output
#

class ModelResponse(BaseModel):
    """Response from a single model extraction."""
    
    model_name: str = Field(
        description="Name of LLM model"
    )
    
    provider: str = Field(
        description="Provider (OpenAI, Anthropic, Google)"
    )
    
    # Essential or Research Extraction
    essential_extraction: Optional[TestPaperExtraction] = Field(
        default=None,
        description="Low-token essential-features output"
    )
    
    extraction: Optional[ResearchPaperExtraction] = Field(
        default=None,
        description="Extracted features"
    )
    
    processing_time: Optional[float] = Field(
        default=None,
        description="Time taken to process in seconds"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if extraction failed"
    )


class ExpertEvaluation(BaseModel):
    """Expert model's evaluation of multiple model responses."""
    
    best_extraction: ResearchPaperExtractionBody = Field(
        description="The best/most accurate extraction after evaluation (no confidence scores)"
    )
    reasoning: str = Field(
        description="Explanation of why this extraction was chosen"
    )
    field_level_decisions: dict[str, str] = Field(
        default_factory=dict,
        description="Per feature decisions on which model's output was chosen"
    )
    
    agreement_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall agreement score between models (0-1)"
    )

class FieldSimilarity(BaseModel):
    """Per-field cosine similarity vs ground truth."""

    field_name: str
    model_value: str
    ground_truth_value: str
    cosine_similarity: float
    target: float
    passed: bool


class SimilarityResult(BaseModel):
    """Aggregated similarity scores (full extraction vs ground truth)."""

    paper_path: str = ""
    string_features: list[FieldSimilarity] = Field(default_factory=list)
    specific_features: list[FieldSimilarity] = Field(default_factory=list)
    mean_string_similarity: Optional[float] = None
    mean_specific_similarity: Optional[float] = None
    mean_overall_similarity: Optional[float] = None
    fields_passing: int = 0
    fields_total: int = 0


class ExtractionResult(BaseModel):
    """Final result containing all model responses and expert evaluation."""
    
    paper_path: str = Field(description="Path to the source PDF")
    
    # One or other.
    essential_features: Optional[TestPaperExtraction] = Field(
        default=None,
        description="Basic features extracted initially"
    )
    
    features: Optional[ResearchPaperExtraction] = Field(
        default=None,
        description="Complete set of extracted features"
    )
    
    model_responses: list[ModelResponse] = Field(
        description="Responses from all models"
    )
    expert_evaluation: Optional[ExpertEvaluation] = Field(
        default=None,
        description="Expert model's evaluation and final decision"
    )
    ground_truth_similarity: Optional[SimilarityResult] = Field(
        default=None,
        description="Cosine similarity vs ground_truth/ JSON when available",
    )
    total_processing_time: Optional[float] = Field(
        default=None,
        description="Total time for complete extraction in seconds"
    )
