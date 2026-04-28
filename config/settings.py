"""
Configuration management.
Notably handles API Keys (from .env), model selection, prompt choice
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class."""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    # OpenRouter (optional; use with --openrouter to bypass direct OpenAI)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    
    # Model Configuration
    OPENAI_MODEL: str = "gpt-5.4"
    OPENAI_MINI_MODEL: str = "gpt-5.4-mini" 
    # Expert judge uses the Chat Completions API; override if your main model is incompatible.
    OPENAI_EXPERT_MODEL: str = "gpt-5.4-mini"
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"
    GOOGLE_MODEL : str = "gemini-2.5-flash"

    # OpenRouter: OpenAI-compatible base URL and model slugs (see https://openrouter.ai/models )
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = "gpt-5.4"
    OPENROUTER_MINI_MODEL: str = "gpt-5.4-mini"
    OPENROUTER_HTTP_REFERER: str = os.getenv("OPENROUTER_HTTP_REFERER", "")
    OPENROUTER_APP_NAME: str = os.getenv("OPENROUTER_APP_NAME", "WikistimProject")
    # Expert judge via OpenRouter (e.g. run_expert_similarity_pipeline.py)
    OPENROUTER_EXPERT_MODEL: str = os.getenv("OPENROUTER_EXPERT_MODEL", "openai/gpt-5.4")
    
    DEFAULT_TEMPERATURE : float = 0.0

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "output"
    LOGS_DIR: Path = BASE_DIR / "logs"
    PROMPTS_DIR: Path = BASE_DIR / "prompts"
    TRAINING_DIR : Path = BASE_DIR / "training_papers"
    ESSENTIALS_DIR : Path = BASE_DIR / "shortened_papers"
    TESTING_DIR : Path = BASE_DIR / "testing_papers"
    GROUND_TRUTH_DIR : Path = BASE_DIR / "ground_truth"
    SIMILARITY_SCORE_DIR: Path = BASE_DIR / "similarity_score"
    # PAPERS_DIR: Path = BASE_DIR / "papers"
    
    # Prompt Selection
    OPENAI_EXPERT_SYSTEM_PROMPT : Path = PROMPTS_DIR / "openai_system_prompt.txt"
    OPENAI_EXPERT_PROMPT : Path = PROMPTS_DIR / "openai_expert_prompt.txt"
    
    OPENAI_SYSTEM_PROMPT : Path = None
    OPENAI_ESSENTIAL_FEATURES_PROMPT : Path = PROMPTS_DIR / "openai_essential_features_prompt.txt"
    OPENAI_FEATURES_PROMPT : Path = PROMPTS_DIR / "openai_features_prompt.txt"
    
    ANTHROPIC_SYSTEM_PROMPT : Path = None
    ANTHROPIC_ESSENTIAL_FEATURES_PROMPT : Path = PROMPTS_DIR / "openai_essential_features_prompt.txt"
    ANTHROPIC_RESEARCH_FEATURES_PROMPT : Path = PROMPTS_DIR / "openai_features_prompt.txt"
    
    GOOGLE_SYSTEM_PROMPT : Path = None 
    GOOGLE_ESSENTIAL_FEATURES_PROMPT : Path = PROMPTS_DIR / "openai_essential_features_prompt.txt"
    GOOGLE_FEATURES_PROMPT : Path = PROMPTS_DIR / "openai_features_prompt.txt"
    
    # Logging Configuration
    LOG_LEVEL: str = 'DEBUG'
    LOG_FILE: Path = LOGS_DIR / 'extraction.log'


    def validate(cls) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY not set")
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY not set")
        if not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY not set")
            
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

    
    def ensure_directories(cls) -> None:
        """Ensure that necessary directories exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.ESSENTIALS_DIR.mkdir(parents=True, exist_ok=True)
        cls.GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
        cls.SIMILARITY_SCORE_DIR.mkdir(parents=True, exist_ok=True)

# use @classmethod or @staticmethod if removed and change imports to Config class instead of object.
config = Config()
