"""Utility for loading prompt texts from the prompts directory."""

from pathlib import Path
from config import config
from models.schemas import TestPaperExtraction

from string import Template

def load_prompt(prompt_path: str) -> str:
    """
    Load a prompt text file.

    Args:
        prompt_path: The path of the prompt file

    Returns:
        str: Text content of the prompt file.

    Raises:
        FileNotFoundError: If prompt file is not found.
    """

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file '{prompt_path}' not found.")
    
    return prompt_path.read_text(encoding="utf-8")

def load_unfilled_features_prompt(essential_features: TestPaperExtraction, unfilled_features_prompt: str) -> str:
    """
    Load the essential features prompt file by populating with text and author to facilitate web search.

    Args:
        essential_features (EssentialFeatures): 
        essential_features_prompt (str): 

    Returns:
        str: Filled prompt text.
    """
    unfilled_prompt = load_prompt(unfilled_features_prompt)
    template = Template(unfilled_prompt)
    
    # Truncate if needed
    title = (essential_features.title or "").strip()
    authors = (essential_features.authors or "").strip()
    
    # Size limiter
    if len(authors) > 1000:
        authors = authors[:1000] + " etc"
    filled_prompt = template.safe_substitute(title=title, authors=authors)
    
    return filled_prompt
