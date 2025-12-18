"""
run_optimization_from_config.py

Run Optuna hyperparameter optimization from YAML config files.

Usage:
  python scripts/run_optimization_from_config.py configs/config_lstm_regression.yaml
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import yaml

from thermocycling.config import RunConfig  # pyright: ignore[reportMissingImports]
from thermocycling.utils.seed import set_global_seed  # pyright: ignore[reportMissingImports]
from thermocycling.pipeline.assemble import full_dataset  # pyright: ignore[reportMissingImports]
from thermocycling.preprocessing.smoothing import Exp_pd, Savitzky_Golay  # pyright: ignore[reportMissingImports]
from thermocycling.models.optuna_objectives import (  # pyright: ignore[reportMissingImports]
    run_lstm_regressor_optimization,
    run_lstm_classifier_optimization,
    run_catboost_regressor_optimization,
    run_catboost_classifier_optimization,
)

DEDRIFT_METHODS = {
    "Exp_pd": Exp_pd,
    "Savitzky_Golay": Savitzky_Golay,
}


def main():
    parser = argparse.ArgumentParser(description="Run optimization from YAML config")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    # Load and validate config
    print(f"📄 Loading config: {args.config}")
    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    
    try:
        cfg = RunConfig.model_validate(raw_config)
    except Exception as e:
        print(f"❌ Config validation failed:\n{e}")
        sys.exit(1)

    # Set global seed
    set_global_seed(cfg.random_seed)

    # Load data
    print(f"\n📊 Preparing data...")
    dedrift_func = DEDRIFT_METHODS[cfg.data.dedrift_method]
    df = full_dataset(dedrifting_func=dedrift_func, **cfg.data.dedrift_params)
    
    print(f"✅ Loaded dataset: {df.shape}")
    print(f"   Gases: {df['gas_'].unique()}")
    print(f"   Cycles: {df['meas_cycle'].min()} - {df['meas_cycle'].max()}")
    print(f"   Target columns: {cfg.data.target_cols}")

    # Build objective kwargs - simplified!
    objective_kwargs = {
        "model_config": cfg.model,  # Pass entire model config dict
        "cv_start_cycle": cfg.preprocessing.cv_start_cycle,
        "cv_test_size": cfg.preprocessing.cv_test_size,
        "feature_cols": cfg.data.feature_cols,
        "target_cols": cfg.data.target_cols,
        "random_seed": cfg.random_seed,
    }

    # Select optimization function
    func_map = {
        ("lstm", "regression"): run_lstm_regressor_optimization,
        ("lstm", "classification"): run_lstm_classifier_optimization,
        ("catboost", "regression"): run_catboost_regressor_optimization,
        ("catboost", "classification"): run_catboost_classifier_optimization,
    }
    
    key = (cfg.experiment.model_type, cfg.experiment.task)
    if key not in func_map:
        print(f"❌ Unknown model_type/task: {key}")
        sys.exit(1)
    
    optimize_func = func_map[key]

    # Run optimization
    print(f"\n🚀 Starting optimization:")
    print(f"   Experiment: {cfg.experiment.name}")
    print(f"   Study: {cfg.experiment.study_name}")
    print(f"   Trials: {cfg.optimization.n_trials}")
    print(f"   Direction: {cfg.optimization.direction}")
    print(f"   Optuna storage: {cfg.storage.optuna}")
    print(f"   MLflow URI: {cfg.mlflow.tracking_uri}")
    print(f"   Results dir: {cfg.results_dir}")

    study = optimize_func(
        df,
        study_name=cfg.experiment.study_name,
        mlflow_experiment_name=cfg.experiment.name,
        n_trials=cfg.optimization.n_trials,
        timeout=cfg.optimization.timeout,
        direction=cfg.optimization.direction,
        objective_kwargs=objective_kwargs,
        storage=cfg.storage.optuna,
        pruner_config=cfg.optimization.pruner.model_dump(),
        mlflow_tracking_uri=cfg.mlflow.tracking_uri,
        results_dir=cfg.results_dir,
        config_dump=raw_config,  # Save original YAML for reproducibility
    )

    print(f"\n✅ Optimization complete!")
    print(f"   Best value: {study.best_value:.6f}")
    print(f"   Best params: {study.best_params}")
    print(f"   Results: {cfg.results_dir}/")


if __name__ == "__main__":
    main()