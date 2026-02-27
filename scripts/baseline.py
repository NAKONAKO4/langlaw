#!/usr/bin/env python3
"""
Baseline script for running PySR without LLM guidance.

Runs symbolic regression using all features as a baseline comparison.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import logging

# Add parent directory to path to import symlaw without installation
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from symlaw.config.settings import Settings
from symlaw.data.loader import five_fold_split, load_all_data, bulk_modulus_preprocessor
from symlaw.models.sr_runner import run_symbolic_regression
from symlaw.utils.logger import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PySR Baseline (All Features, No LLM)",
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
        help='Specific fold to run. If not provided, runs all folds.'
    )
    
    parser.add_argument(
        '--all-data',
        action='store_true',
        help='Train on entire dataset without split (ignores --fold)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='Baseline',
        help='Name for this baseline run'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(name="symlaw.baseline", level=logging.INFO)
    
    logger.info("="*50)
    logger.info("SymLaw Baseline - Full Feature Set")
    logger.info("="*50)
    
    # Load settings
    try:
        settings = Settings.from_yaml(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_results_dir = os.path.join(settings.data.results_dir, f"{args.name}_{timestamp}")
    
    if args.all_data:
        run_results_dir += "_AllData"
    elif args.fold is not None:
        run_results_dir += f"_fold{args.fold}"
    
    os.makedirs(run_results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {run_results_dir}")
    
    # Baseline PySR parameters (simpler for baseline)
    baseline_params = {
        'niterations': 100,
        'maxdepth': 16,
    }
    
    all_run_results = {}
    all_fold_maes = []
    
    # Determine preprocessor
    preprocessor = bulk_modulus_preprocessor if settings.data.target == 'B0_eV_A3' else None
    
    # Run based on mode
    if args.all_data:
        logger.info("Running on full dataset (no train/test split)")
        X, y = load_all_data(
            settings.data.data_path,
            settings.data.all_features,
            settings.data.target,
            preprocessor=preprocessor
        )
        
        if X is None:
            logger.error("Failed to load data. Exiting.")
            return
        
        # Use same data for train and test (training score)
        sr_results = run_symbolic_regression(X, y, X, y, settings, llm_params=baseline_params)
        
        logger.info(f"Training MAE: {sr_results['mae']:.4f}, R²: {sr_results['r2']:.4f}")
        
        all_run_results['full_data'] = [{
            "mode": "Full Data",
            "features_used": settings.data.all_features,
            "mae": sr_results['mae'],
            "r2": sr_results['r2'],
            "rmse": sr_results['rmse'],
            "formula": sr_results['formula']
        }]
        
    else:
        n_folds = settings.experiment.n_folds
        folds_to_run = [args.fold] if args.fold is not None else range(n_folds)
        
        for fold_idx in folds_to_run:
            logger.info(f"\n{'*'*20} Running Fold {fold_idx} {'*'*20}")
            
            X_train, X_test, y_train, y_test = five_fold_split(
                settings.data.data_path,
                settings.data.all_features,
                settings.data.target,
                fold_index=fold_idx,
                n_splits=n_folds,
                preprocessor=preprocessor
            )
            
            if X_train is None:
                logger.error(f"Failed to load data for fold {fold_idx}")
                continue
            
            sr_results = run_symbolic_regression(
                X_train, y_train, X_test, y_test,
                settings, llm_params=baseline_params
            )
            
            logger.info(f"Fold {fold_idx} - MAE: {sr_results['mae']:.4f}, R²: {sr_results['r2']:.4f}")
            
            fold_key = f"fold_{fold_idx}"
            all_run_results[fold_key] = [{
                "fold": fold_idx,
                "features_used": settings.data.all_features,
                "mae": sr_results['mae'],
                "r2": sr_results['r2'],
                "rmse": sr_results['rmse'],
                "formula": sr_results['formula']
            }]
            
            all_fold_maes.append(sr_results['mae'])
            
            # Save fold report
            report_path = os.path.join(run_results_dir, f"fold_{fold_idx}_results.log")
            with open(report_path, 'w') as f:
                f.write(f"Baseline Run - Fold {fold_idx}\n")
                f.write(f"Features: {settings.data.all_features}\n\n")
                f.write(f"MAE: {sr_results['mae']}\n")
                f.write(f"RMSE: {sr_results['rmse']}\n")
                f.write(f"R²: {sr_results['r2']}\n")
                f.write(f"Formula: {sr_results['formula']}\n")
    
    # Save summary
    summary_path = os.path.join(run_results_dir, "summary_results.json")
    with open(summary_path, 'w') as f:
        json.dump(all_run_results, f, indent=4)
    logger.info(f"Summary saved to {summary_path}")
    
    if all_fold_maes:
        import numpy as np
        logger.info(f"\nAverage MAE across folds: {np.mean(all_fold_maes):.6f}")


if __name__ == "__main__":
    main()
