"""
LLM-based feature selection for symbolic regression.

Uses LLM to intelligently select features and suggest PySR parameters
based on experience pool and task-specific prompts.
"""

from openai import OpenAI
import json
import re
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from symlaw.config.settings import Settings

logger = logging.getLogger("symlaw.llm")


def load_prompt_template(
    filename: str,
    all_features: List[str],
    experience_pool: str,
    prompts_dir: Optional[str] = None
) -> Optional[str]:
    """
    Load and format a prompt template from file.
    
    Args:
        filename: Name of the prompt file
        all_features: List of all available features
        experience_pool: JSON string of experience pool
        prompts_dir: Directory containing prompt files. If None, uses ./prompts
        
    Returns:
        Formatted prompt string or None on error
    """
    if prompts_dir is None:
        # Default to prompts/ directory relative to project root
        current_dir = Path(__file__).parent.parent.parent
        prompts_dir = current_dir / "prompts"
    else:
        prompts_dir = Path(prompts_dir)
    
    file_path = prompts_dir / filename
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Format template with dynamic content
        # Note: {{...}} in template becomes {...} (for JSON examples)
        formatted = template.format(
            all_features=all_features,
            experience_pool=experience_pool
        )
        
        return formatted
        
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        return None
    except KeyError as e:
        logger.error(f"Missing placeholder in prompt template: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading prompt template: {e}")
        return None


def get_features_from_llm(
    settings: Settings,
    experience_pool_str: str,
    prompts_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Query LLM for feature selection and PySR parameter suggestions.
    
    Args:
        settings: Settings instance with LLM and data configuration
        experience_pool_str: JSON string of previous experiments
        prompts_dir: Optional directory containing prompt files
        
    Returns:
        Dictionary containing:
            - features_library: List of selected features
            - pysr_params: Dict of suggested PySR parameters
            - reasoning: LLM's reasoning for the selection
        Returns None on error
        
    Example:
        >>> suggestion = get_features_from_llm(settings, experience_pool)
        >>> if suggestion:
        ...     features = suggestion['features_library']
        ...     params = suggestion['pysr_params']
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
    )
    
    # System prompt
    sys_prompt = """
    You are an expert AI assistant specializing in materials science and data science.
    Your core mission is to assist in discovering simple, physically meaningful symbolic equations from computational materials data using PySR.
    You MUST ONLY output a single, valid JSON object. Do not add any extra text, explanation, or markdown formatting like ```json.
    """
    
    # Select prompt file based on target
    target_prompt_map = {
        'B0_eV_A3': 'bulk_modulus.txt',
        'Bandgap': 'band_gap.txt',
        'CB_dir': 'band_gap.txt',
        'epsilon': 'solvent.txt',
        'VRHE': 'oer.txt'  # Assuming you'll create this
    }
    
    target = settings.data.target
    prompt_file = target_prompt_map.get(target)
    
    if not prompt_file:
        logger.error(f"No prompt template found for target '{target}'")
        logger.warning(f"Available targets: {list(target_prompt_map.keys())}")
        return None
    
    # Load and format user prompt
    user_prompt = load_prompt_template(
        prompt_file,
        all_features=settings.data.all_features,
        experience_pool=experience_pool_str,
        prompts_dir=prompts_dir
    )
    
    if user_prompt is None:
        return None
    
    # Call LLM
    try:
        logger.info(f"Requesting feature selection from LLM (Target: {target}, Model: {settings.llm.model_name})...")
        
        response = client.chat.completions.create(
            model=settings.llm.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_body={"thinking_mode": False},
        )
        
        response_content = response.choices[0].message.content
        logger.debug(f"LLM response received: {response_content[:200]}...")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            llm_suggestion = json.loads(json_match.group(0))
            
            # Validate response structure
            required_keys = ['features_library', 'pysr_params', 'reasoning']
            missing_keys = [k for k in required_keys if k not in llm_suggestion]
            
            if missing_keys:
                logger.error(f"LLM response missing required keys: {missing_keys}")
                return None
            
            logger.info("Successfully received LLM suggestion")
            return llm_suggestion
        else:
            logger.error("No valid JSON found in LLM response")
            logger.debug(f"Full response: {response_content}")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        return None
