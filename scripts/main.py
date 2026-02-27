#!/usr/bin/env python3
"""
Main script for running SymLaw - LLM-guided symbolic regression.

This script orchestrates the full workflow: data loading, LLM feature selection,
symbolic regression, and results logging across multiple folds.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

# Add parent directory to path to import symlaw without installation
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from symlaw.config.settings import Settings, set_settings
from symlaw.data.loader import five_fold_split, bulk_modulus_preprocessor
from symlaw.models.llm_selector import get_features_from_llm
from symlaw.models.sr_runner import run_symbolic_regression
from symlaw.utils.logger import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SymLaw - LLM-Guided Symbolic Regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/bulk_modulus.yaml',
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='Specific fold index to run (0 to n_folds-1). If not provided, runs all folds.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='LLM model name to use (overrides config file)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def get_experience_pool_path(base_path: str, fold_idx: int) -> str:
    """Get fold-specific experience pool path."""
    base_dir = os.path.dirname(base_path)
    base_filename = os.path.basename(base_path)
    name_part, ext_part = os.path.splitext(base_filename)
    return os.path.join(base_dir, f"{name_part}_fold_{fold_idx}{ext_part}")


def load_experience_pool(pool_path: str, fold_idx: int, logger: logging.Logger) -> List[dict]:
    """Load experience pool for a specific fold."""
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
        logger.info(f"Loaded {len(pool)} experiences from {pool_path}")
        return pool
    else:
        logger.info(f"No experience pool found for Fold {fold_idx}. Starting fresh.")
        return []


def save_experience_pool(pool: List[dict], pool_path: str, logger: logging.Logger) -> None:
    """Save experience pool to file."""
    os.makedirs(os.path.dirname(pool_path), exist_ok=True)
    with open(pool_path, 'w') as f:
        json.dump(pool, f, indent=4)
    logger.debug(f"Experience pool saved to {pool_path}")


def format_experience_pool_str(pool: List[dict], max_experiences: int) -> str:
    """Format experience pool for LLM prompt."""
    if not pool:
        return "The Experience Pool is empty. This is the first run."
    
    sorted_pool = sorted(pool, key=lambda x: x.get('test_set_mae', float('inf')))
    top_experiences = sorted_pool[:max_experiences]
    return json.dumps(top_experiences, indent=2)


def run_single_fold(
    settings: Settings,
    fold_idx: int,
    run_results_dir: str,
    model_prefix: str,
    logger: logging.Logger
) -> dict:
    """
    Run all rounds for a single fold.
    
    Returns:
        Dictionary with fold results
    """
    logger.info(f"{'*'*25} STARTING FOLD {fold_idx} / {settings.experiment.n_folds} {'*'*25}")
    
    fold_results = []
    
    # Load experience pool
    current_fold_pool_path = get_experience_pool_path(
        settings.data.experience_pool_path, fold_idx
    )
    experience_pool = load_experience_pool(current_fold_pool_path, fold_idx, logger)
    
    # Load data for this fold
    # Use bulk_modulus_preprocessor if target is bulk modulus
    preprocessor = bulk_modulus_preprocessor if settings.data.target == 'B0_eV_A3' else None
    
    X_train, X_test, y_train, y_test = five_fold_split(
        settings.data.data_path,
        settings.data.all_features,
        settings.data.target,
        fold_index=fold_idx,
        n_splits=settings.experiment.n_folds,
        preprocessor=preprocessor
    )
    
    if X_train is None:
        logger.error(f"Failed to load data for fold {fold_idx}. Skipping this fold.")
        return {'fold': fold_idx, 'results': []}
    
    # Run rounds
    start_round = len(experience_pool)
    for i in range(start_round, start_round + settings.experiment.num_rounds):
        logger.info(f"{'*'*20} FOLD {fold_idx} / ROUND {i + 1} {'*'*20}")
        
        # Format experience pool for LLM
        experience_pool_str = format_experience_pool_str(
            experience_pool, settings.experiment.max_experiences_in_prompt
        )
        
        # Get LLM suggestion
        llm_suggestion = get_features_from_llm(settings, experience_pool_str)
        if not llm_suggestion or 'features_library' not in llm_suggestion or 'pysr_params' not in llm_suggestion:
            logger.warning("Failed to get valid LLM suggestion. Skipping this round.")
            continue
        
        selected_features = llm_suggestion['features_library']
        llm_pysr_params = llm_suggestion['pysr_params']
        llm_reasoning = llm_suggestion['reasoning']
        
        logger.info(f"LLM Model: {model_prefix}")
        logger.info(f"LLM Reasoning: {llm_reasoning}")
        logger.info(f"LLM suggested features: {selected_features}")
        
        # Validate features
        try:
            X_train_subset = X_train[selected_features]
            X_test_subset = X_test[selected_features]
        except KeyError as e:
            logger.error(f"LLM suggested invalid feature: {e}. Skipping round.")
            continue
        
        # Run symbolic regression
        sr_results = run_symbolic_regression(
            X_train_subset, y_train, X_test_subset, y_test,
            settings, llm_params=llm_pysr_params
        )
        
        logger.info(f"Round {i + 1} Results → R²: {sr_results['r2']:.4f}, MAE: {sr_results['mae']:.4f}")
        
        # Prepare equations list
        equations_df = sr_results['equations_df']
        if not equations_df.empty:
            equations_list = equations_df[['complexity', 'loss', 'score', 'equation']].to_dict(orient='records')
        else:
            equations_list = []
        
        # Log current round
        current_round_log = {
            "round": i + 1,
            "fold": fold_idx,
            "model": model_prefix,
            "llm_reasoning": llm_reasoning,
            "features_used": selected_features,
            "pysr_params_used": llm_pysr_params,
            "test_set_r2_score": sr_results['r2'],
            "test_set_mae": sr_results['mae'],
            "test_set_rmse": sr_results['rmse'],
            "latex": str(sr_results['latex']),
            "formula": sr_results['formula'],
            "equations_table": equations_list
        }
        
        # Update experience pool
        exp_log = {
            'round': i + 1,
            'features_used': selected_features,
            'pysr_params_used': llm_pysr_params,
            'test_set_mae': sr_results['mae'],
            'formula': sr_results['formula'],
        }
        experience_pool.append(exp_log)
        save_experience_pool(experience_pool, current_fold_pool_path, logger)
        
        fold_results.append(current_round_log)
        
        # Save detailed round report
        round_result_filename = os.path.join(
            run_results_dir, f"fold_{fold_idx}_round_{i+1}_results.log"
        )
        with open(round_result_filename, 'w') as f:
            f.write(f"Model Used: {model_prefix}\n")
            f.write(f"{'*'*10}Fold {fold_idx} / Round {i+1} Detailed Report{'*'*10}\n\n")
            f.write(f"LLM Reasoning:\n{llm_reasoning}\n\n")
            f.write(f"Features Used: {selected_features}\n")
            f.write(f"PySR Params Used: {llm_pysr_params}\n\n")
            f.write(f"Test Set MAE: {sr_results['mae']}\n")
            f.write(f"Test Set RMSE: {sr_results['rmse']}\n")
            f.write(f"Formula: {sr_results['formula']}\n")
            f.write(f"{'*'*10}PySR Equations DataFrame{'*'*10}\n")
            f.write(sr_results['equations_df'].to_string())
        logger.info(f"Detailed report saved to {round_result_filename}")
    
    # Compute best MAE for this fold
    if fold_results:
        best_mae = min(r['test_set_mae'] for r in fold_results)
        logger.info(f"{'='*20} FOLD {fold_idx} BEST MAE: {best_mae:.6f} {'='*20}")
    
    return {'fold': fold_idx, 'results': fold_results}


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logger
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(name="symlaw", level=log_level)
    
    logger.info("="*50)
    logger.info("SymLaw - LLM-Guided Symbolic Regression")
    logger.info("="*50)
    
    # Load settings
    try:
        settings = Settings.from_yaml(args.config)
        set_settings(settings)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Override model if specified
    if args.model:
        logger.info(f"Overriding LLM model: {args.model}")
        settings.llm.model_name = args.model
        model_prefix = args.model
    else:
        model_prefix = settings.llm.model_name
    
    # Determine folds to run
    n_folds = settings.experiment.n_folds
    if args.fold is not None:
        if 0 <= args.fold < n_folds:
            folds_to_run = [args.fold]
            logger.info(f"Running single fold: {args.fold}")
        else:
            logger.error(f"Fold index {args.fold} out of range (0-{n_folds-1})")
            return
    else:
        folds_to_run = range(n_folds)
        logger.info(f"Running all {n_folds} folds")
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_results_dir = os.path.join(settings.data.results_dir, f"{model_prefix}_{timestamp}")
    if args.fold is not None:
        run_results_dir += f"_fold{args.fold}"
    
    os.makedirs(run_results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {run_results_dir}")
    
    # Run folds
    all_run_results = {}
    all_fold_best_maes = []
    
    for fold_idx in folds_to_run:
        fold_result = run_single_fold(settings, fold_idx, run_results_dir, model_prefix, logger)
        fold_key = f"fold_{fold_idx}"
        all_run_results[fold_key] = fold_result['results']
        
        if fold_result['results']:
            best_mae = min(r['test_set_mae'] for r in fold_result['results'])
            all_fold_best_maes.append(best_mae)
    
    # Save summary
    summary_file_path = os.path.join(run_results_dir, "summary_all_folds.json")
    with open(summary_file_path, 'w') as f:
        json.dump(all_run_results, f, indent=4)
    logger.info(f"Complete run summary saved to {summary_file_path}")
    
    # Final statistics
    if all_fold_best_maes:
        import numpy as np
        avg_best_mae = np.mean(all_fold_best_maes)
        logger.info(f"\n{'='*50}")
        logger.info(f"Average Best MAE (across executed folds): {avg_best_mae:.6f}")
        logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
