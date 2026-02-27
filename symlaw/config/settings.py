"""
Unified configuration management system for SymLaw.

This module provides a centralized configuration system using Pydantic for validation
and supporting both environment variables and YAML configuration files.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class PySRConfig(BaseModel):
    """PySR symbolic regression configuration."""
    
    niterations: int = Field(default=500, description="Number of iterations")
    binary_operators: List[str] = Field(default=['+', '-', '*', '/', '^'], description="Binary operators")
    model_selection: str = Field(default='best', description="Model selection strategy")
    random_state: int = Field(default=42, description="Random seed")
    elementwise_loss: str = Field(default="L1DistLoss()", description="Loss function")
    maxdepth: int = Field(default=10, description="Maximum depth")
    turbo: bool = Field(default=True, description="Turbo mode")
    constraints: Dict[str, tuple] = Field(default={'^': (-1, 1)}, description="Operator constraints")


class LLMConfig(BaseModel):
    """LLM API configuration."""
    
    api_key: str = Field(..., description="LLM API key")
    base_url: str = Field(default="https://chat.intern-ai.org.cn/api/v1/", description="LLM base URL")
    model_name: str = Field(default="intern-s1", description="LLM model name")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty."""
        if not v or v == "your-api-key-here":
            raise ValueError("LLM_API_KEY must be set in environment variables or config file")
        return v


class DataConfig(BaseModel):
    """Data and feature configuration."""
    
    data_path: str = Field(..., description="Path to data CSV file")
    experience_pool_path: str = Field(..., description="Path to experience pool JSON")
    results_dir: str = Field(default="results", description="Results output directory")
    all_features: List[str] = Field(..., description="All available features")
    target: str = Field(..., description="Target variable name")


class ExperimentConfig(BaseModel):
    """Experiment runtime configuration."""
    
    num_rounds: int = Field(default=20, description="Number of rounds per fold")
    max_experiences_in_prompt: int = Field(default=15, description="Max experiences to include in LLM prompt")
    n_folds: int = Field(default=5, description="Number of cross-validation folds")


class Settings(BaseModel):
    """Main settings container for SymLaw."""
    
    llm: LLMConfig
    data: DataConfig
    pysr: PySRConfig = Field(default_factory=PySRConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Settings":
        """
        Load settings from a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Settings instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Override with environment variables if set
        if "llm" not in config_data:
            config_data["llm"] = {}
        
        config_data["llm"]["api_key"] = os.getenv("LLM_API_KEY", config_data["llm"].get("api_key", ""))
        config_data["llm"]["base_url"] = os.getenv("LLM_BASE_URL", config_data["llm"].get("base_url", "https://chat.intern-ai.org.cn/api/v1/"))
        config_data["llm"]["model_name"] = os.getenv("LLM_MODEL_NAME", config_data["llm"].get("model_name", "intern-s1"))
        
        return cls(**config_data)
    
    @classmethod
    def from_legacy_config(cls, config_module) -> "Settings":
        """
        Create Settings from legacy config.py module for backward compatibility.
        
        Args:
            config_module: Imported config module
            
        Returns:
            Settings instance
        """
        return cls(
            llm=LLMConfig(
                api_key=getattr(config_module, "LLM_API_KEY", os.getenv("LLM_API_KEY", "")),
                base_url=getattr(config_module, "LLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/"),
                model_name=getattr(config_module, "LLM_MODEL_NAME", "intern-s1")
            ),
            data=DataConfig(
                data_path=config_module.DATA_PATH,
                experience_pool_path=config_module.EXPERIENCE_POOL_PATH,
                results_dir=config_module.RESULTS_DIR,
                all_features=config_module.ALL_FEATURES,
                target=config_module.TARGET
            ),
            pysr=PySRConfig(**config_module.PYSR_CONFIG) if hasattr(config_module, "PYSR_CONFIG") else PySRConfig(),
            experiment=ExperimentConfig(
                num_rounds=getattr(config_module, "NUM_ROUNDS", 20),
                max_experiences_in_prompt=getattr(config_module, "MAX_EXPERIENCES_IN_PROMPT", 15)
            )
        )
    
    def get_pysr_params(self, llm_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get PySR parameters as dictionary, optionally merged with LLM suggestions.
        
        Args:
            llm_override: Optional LLM-suggested parameters to override defaults
            
        Returns:
            Dictionary of PySR parameters
        """
        params = self.pysr.model_dump()
        if llm_override:
            params.update(llm_override)
        return params


# Singleton instance for backward compatibility
_settings: Optional[Settings] = None


def get_settings() -> Optional[Settings]:
    """Get the current global settings instance."""
    return _settings


def set_settings(settings: Settings) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings
