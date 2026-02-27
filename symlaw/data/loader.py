"""
Data loading and preprocessing utilities for SymLaw.

Provides clean, reusable functions for loading datasets, performing train/test splits,
and K-fold cross-validation with optional data preprocessing pipelines.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Callable
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import logging

logger = logging.getLogger("symlaw.data")


def clean_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    strip_brackets: bool = True
) -> pd.DataFrame:
    """
    Clean and convert columns to numeric type.
    
    Args:
        df: Input DataFrame
        columns: List of column names to clean
        strip_brackets: If True, strip square brackets from string values
        
    Returns:
        DataFrame with cleaned numeric columns
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
            
        if df_clean[col].dtype == 'object' and strip_brackets:
            df_clean[col] = df_clean[col].str.strip('[]')
        
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        nan_count = df_clean[col].isna().sum()
        if nan_count > 0:
            logger.debug(f"Column '{col}': {nan_count} values converted to NaN")
    
    return df_clean


def load_and_split_data(
    data_path: str,
    feature_list: List[str],
    target: str,
    test_size: float = 0.02,
    random_state: int = 42,
    drop_na: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], 
           Optional[pd.Series], Optional[pd.Series]]:
    """
    Load data from CSV and split into training and testing sets.
    
    Args:
        data_path: Path to CSV file
        feature_list: List of feature column names
        target: Target column name
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        drop_na: If True, drop rows with missing values after cleaning
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) or (None, None, None, None) on error
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data from {data_path} ({len(df)} rows)")
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None
    
    # Validate columns exist
    missing_cols = set(feature_list + [target]) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns in data: {missing_cols}")
        return None, None, None, None
    
    X = df[feature_list].copy()
    y = df[target].copy()
    
    # Clean and convert features
    logger.debug("Cleaning and converting data to numeric...")
    X = clean_numeric_columns(X, feature_list)
    
    # Handle missing values
    if drop_na:
        combined_data = pd.concat([X, y], axis=1)
        original_rows = len(combined_data)
        combined_data.dropna(inplace=True)
        cleaned_rows = len(combined_data)
        
        if original_rows > cleaned_rows:
            logger.info(f"Dropped {original_rows - cleaned_rows} rows with missing values")
        
        X = combined_data[feature_list]
        y = combined_data[target]
    
    if X.empty:
        logger.error("No data remaining after cleaning")
        return None, None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def five_fold_split(
    data_path: str,
    feature_list: List[str],
    target: str,
    fold_index: int,
    n_splits: int = 5,
    random_state: int = 42,
    preprocessor: Optional[Callable[[pd.DataFrame, List[str]], pd.DataFrame]] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame],
           Optional[pd.Series], Optional[pd.Series]]:
    """
    Perform K-fold cross-validation split and return data for a specific fold.
    
    Args:
        data_path: Path to CSV file
        feature_list: List of feature column names
        target: Target column name
        fold_index: Current fold index (0 to n_splits-1)
        n_splits: Total number of folds
        random_state: Random seed for reproducibility
        preprocessor: Optional function to preprocess features DataFrame
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) for the specified fold,
        or (None, None, None, None) on error
        
    Example:
        >>> X_train, X_test, y_train, y_test = five_fold_split(
        ...     "data.csv", ["feat1", "feat2"], "target", fold_index=0
        ... )
    """
    # Validate fold index
    if not (0 <= fold_index < n_splits):
        logger.error(f"fold_index must be between 0 and {n_splits-1}, got {fold_index}")
        return None, None, None, None
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path} ({len(df)} rows)")
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None
    
    # Validate columns
    missing_cols = set(feature_list + [target]) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns in data: {missing_cols}")
        return None, None, None, None
    
    X = df[feature_list].copy()
    y = df[target].copy()
    
    # Clean numeric columns
    X = clean_numeric_columns(X, feature_list)
    
    # Apply custom preprocessor if provided
    if preprocessor:
        logger.debug("Applying custom preprocessor...")
        X = preprocessor(X, feature_list)
    
    # Drop rows with missing values
    combined_data = pd.concat([X, y], axis=1)
    combined_data.dropna(inplace=True)
    
    X_cleaned = combined_data[feature_list]
    y_cleaned = combined_data[target]
    
    if X_cleaned.empty:
        logger.error("No data remaining after cleaning")
        return None, None, None, None
    
    # Perform K-fold split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_splits = list(kf.split(X_cleaned))
    train_index, test_index = all_splits[fold_index]
    
    X_train = X_cleaned.iloc[train_index]
    X_test = X_cleaned.iloc[test_index]
    y_train = y_cleaned.iloc[train_index]
    y_test = y_cleaned.iloc[test_index]
    
    logger.info(f"Fold {fold_index+1}/{n_splits}: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def load_all_data(
    data_path: str,
    feature_list: List[str],
    target: str,
    preprocessor: Optional[Callable[[pd.DataFrame, List[str]], pd.DataFrame]] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load all data without train/test split (useful for final model training).
    
    Args:
        data_path: Path to CSV file
        feature_list: List of feature column names
        target: Target column name
        preprocessor: Optional function to preprocess features DataFrame
        
    Returns:
        Tuple of (X, y) or (None, None) on error
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data from {data_path} ({len(df)} rows)")
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None
    
    # Validate columns
    missing_cols = set(feature_list + [target]) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns in data: {missing_cols}")
        return None, None
    
    X = df[feature_list].copy()
    y = df[target].copy()
    
    # Clean numeric columns
    X = clean_numeric_columns(X, feature_list)
    
    # Apply custom preprocessor if provided
    if preprocessor:
        logger.debug("Applying custom preprocessor...")
        X = preprocessor(X, feature_list)
    
    # Drop rows with missing values
    combined_data = pd.concat([X, y], axis=1)
    original_rows = len(combined_data)
    combined_data.dropna(inplace=True)
    cleaned_rows = len(combined_data)
    
    if original_rows > cleaned_rows:
        logger.info(f"Dropped {original_rows - cleaned_rows} rows with missing values")
    
    X_cleaned = combined_data[feature_list]
    y_cleaned = combined_data[target]
    
    if X_cleaned.empty:
        logger.error("No data remaining after cleaning")
        return None, None
    
    logger.info(f"Successfully cleaned data: {len(X_cleaned)} samples")
    return X_cleaned, y_cleaned


# Example preprocessor for bulk modulus dataset
def bulk_modulus_preprocessor(X: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    Preprocessing specific to bulk modulus perovskite dataset.
    
    Handles unit conversions and derived features for the ABO₃ dataset.
    """
    X = X.copy()
    
    # Define columns that need unit conversion (eV to J or similar)
    conversion_factor = 1.602e-19
    cols_to_convert = ['epsilon_L_B', 'epsilon_H_B', 'EA_A', 'EA_B', 'IP_A', 'IP_B']
    
    for col in cols_to_convert:
        if col in X.columns:
            X[col] = X[col] / conversion_factor
            logger.debug(f"Converted {col} by factor {conversion_factor}")
    
    # Recalculate electronegativity (EN) from EA and IP
    if 'EA_A' in X.columns and 'IP_A' in X.columns:
        X['EN_A'] = (X['EA_A'] + X['IP_A']) / 2
        logger.debug("Recalculated EN_A from EA_A and IP_A")
    
    if 'EA_B' in X.columns and 'IP_B' in X.columns:
        X['EN_B'] = (X['EA_B'] + X['IP_B']) / 2
        logger.debug("Recalculated EN_B from EA_B and IP_B")
    
    return X
