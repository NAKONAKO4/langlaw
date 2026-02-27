"""
Symbolic regression runner using PySR.

Handles the execution of symbolic regression and metrics computation.
"""

from pysr import PySRRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from symlaw.config.settings import Settings

logger = logging.getLogger("symlaw.sr")


def run_symbolic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    settings: Settings,
    llm_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run symbolic regression using PySR.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target Series
        X_test: Test features DataFrame
        y_test: Test target Series
        settings: Settings instance with PySR configuration
        llm_params: Optional LLM-suggested parameters to override defaults
        
    Returns:
        Dictionary containing:
            - formula: LaTeX formula string
            - latex: LaTeX callable
            - equations_df: DataFrame of all discovered equations
            - r2: R² score on test set
            - mae: Mean absolute error on test set
            - rmse: Root mean squared error on test set
            
    Example:
        >>> results = run_symbolic_regression(X_train, y_train, X_test, y_test, settings)
        >>> print(f"Best formula: {results['formula']}")
        >>> print(f"Test MAE: {results['mae']:.4f}")
    """
    # Get PySR parameters with optional LLM overrides
    pysr_params = settings.get_pysr_params(llm_override=llm_params)
    
    logger.info(f"Starting symbolic regression with {len(X_train.columns)} features: {list(X_train.columns)}")
    logger.debug(f"PySR parameters: {pysr_params}")
    
    # Initialize and fit PySR model
    try:
        model = PySRRegressor(**pysr_params)
        model.fit(X_train, y_train)
        logger.info("Symbolic regression completed successfully")
    except Exception as e:
        logger.error(f"Error during symbolic regression: {e}")
        # Return default error results
        return {
            'formula': "Error during fitting",
            'latex': lambda: "Error",
            'equations_df': pd.DataFrame(),
            'r2': 0.0,
            'mae': np.inf,
            'rmse': np.inf
        }
    
    # Compute metrics on test set
    r2 = 0.0
    mae = np.inf
    rmse = np.inf
    
    try:
        y_pred = model.predict(X_test)
        
        # Only compute metrics if predictions are valid
        if len(y_test.unique()) > 1 and not np.isnan(y_pred).any():
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            temp_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse = np.inf if np.isnan(temp_rmse) else temp_rmse
            
            logger.info(f"Test metrics - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        else:
            logger.warning("Predictions contain NaN or target has no variance, metrics set to defaults")
            r2 = 0.0
            mae = np.inf
            rmse = np.inf
            
    except Exception as e:
        logger.error(f"Error during prediction or metric computation: {e}")
        r2 = 0.0
        mae = np.inf
        rmse = np.inf
    
    # Prepare results
    results = {
        'formula': model.latex() if hasattr(model, 'latex') else "N/A",
        'latex': model.latex if hasattr(model, 'latex') else lambda: "N/A",
        'equations_df': model.equations_ if hasattr(model, 'equations_') else pd.DataFrame(),
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }
    
    return results
