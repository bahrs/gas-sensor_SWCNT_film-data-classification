"""
run_optimization_from_config.py

Run Optuna hyperparameter optimization from YAML config files.
Now supports SQLite storage and pruning configuration.

Usage:
    python scripts/run_optimization_from_config.py configs/lstm_regression.yaml
"""

import sys
import yaml
import argparse
import mlflow
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.assemble import full_dataset  # pyright: ignore[reportMissingImports]
from preprocessing.smoothing import Exp_pd, Savitzky_Golay  # pyright: ignore[reportMissingImports]
from models.optuna_objectives import (  # pyright: ignore[reportMissingImports]
    run_lstm_regressor_optimization,
    run_lstm_classifier_optimization,
    run_catboost_classifier_optimization,
    run_catboost_regressor_optimization
)


DEDRIFT_METHODS = {
    'Exp_pd': Exp_pd,
    'Savitzky_Golay': Savitzky_Golay
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """Prepare dataset according to config."""
    data_cfg = config['data']
    
    # Get dedrift method
    dedrift_method_name = data_cfg['dedrift_method']
    dedrift_func = DEDRIFT_METHODS[dedrift_method_name]
    dedrift_params = data_cfg['dedrift_params']
    
    # Load full dataset
    df = full_dataset(
        dedrifting_func=dedrift_func,
        **dedrift_params
    )
    
    print(f"✓ Loaded dataset: {df.shape}")
    print(f"  Gases: {df['gas_'].unique()}")
    print(f"  Measurement cycles: {df['meas_cycle'].min()} - {df['meas_cycle'].max()}")
    
    return df


def run_optimization(config: dict, df):
    """Run optimization based on config."""
    exp_cfg = config['experiment']
    opt_cfg = config['optimization']
    preproc_cfg = config['preprocessing']
    storage_cfg = config.get('storage', {})
    
    # Get storage paths
    optuna_storage = storage_cfg.get('optuna', 'sqlite:///optuna_studies.db')
    mlflow_storage = storage_cfg.get('mlflow', 'sqlite:///mlflow.db')
    
    # Prepare objective kwargs
    objective_kwargs = {
        'cv_start_cycle': preproc_cfg['cv_start_cycle'],
        'cv_test_size': preproc_cfg['cv_test_size'],
        'feature_cols': config['data']['feature_cols']
    }
    
    # Add model-specific parameters
    if exp_cfg['model_type'] == 'lstm':
        objective_kwargs.update({
            'look_back_range': preproc_cfg['look_back_range'],
            'n_components_range': preproc_cfg['n_components_range']
        })
        
        if exp_cfg['task'] == 'regression':
            objective_kwargs['target_cols'] = config['data']['target_cols']
    
    elif exp_cfg['model_type'] == 'catboost':
        objective_kwargs['n_components_range'] = preproc_cfg['n_components_range']
        
        if exp_cfg['task'] == 'regression':
            objective_kwargs['target_cols'] = config['data']['target_cols']
    
    # Get pruner config
    pruner_cfg = opt_cfg.get('pruner', {})
    pruner_config = {
        'n_startup_trials': pruner_cfg.get('n_startup_trials', 5),
        'n_warmup_steps': pruner_cfg.get('n_warmup_steps', 1),
        'interval_steps': pruner_cfg.get('interval_steps', 1)
    }
    
    # Select optimization function
    if exp_cfg['model_type'] == 'lstm' and exp_cfg['task'] == 'regression':
        optimize_func = run_lstm_regressor_optimization
    elif exp_cfg['model_type'] == 'lstm' and exp_cfg['task'] == 'classification':
        optimize_func = run_lstm_classifier_optimization
    elif exp_cfg['model_type'] == 'catboost' and exp_cfg['task'] == 'classification':
        optimize_func = run_catboost_classifier_optimization
    elif exp_cfg['model_type'] == 'catboost' and exp_cfg['task'] == 'regression':
        optimize_func = run_catboost_regressor_optimization
    else:
        raise ValueError(
            f"Unknown model_type/task combination: "
            f"{exp_cfg['model_type']}/{exp_cfg['task']}")
    
    # Run optimization
    print(f"\n🚀 Starting optimization: {exp_cfg['name']}")
    print(f"   Study: {exp_cfg['study_name']}")
    print(f"   Trials: {opt_cfg['n_trials']}")
    print(f"   Direction: {opt_cfg['direction']}")
    print(f"   Optuna storage: {optuna_storage}")
    print(f"   MLflow storage: {mlflow_storage}")
    print(f"   Pruner: {pruner_cfg.get('type', 'MedianPruner')}")
    print(f"     - startup_trials: {pruner_config['n_startup_trials']}")
    print(f"     - warmup_steps: {pruner_config['n_warmup_steps']}")
    
    study = optimize_func(
        df,
        study_name=exp_cfg['study_name'],
        mlflow_experiment_name=exp_cfg['name'],
        n_trials=opt_cfg['n_trials'],
        timeout=opt_cfg.get('timeout'),
        direction=opt_cfg['direction'],
        objective_kwargs=objective_kwargs,
        storage=optuna_storage,
        pruner_config=pruner_config
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best value: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")
    
    # Print pruning statistics
    n_complete = len([t for t in study.trials if t.state.name == 'COMPLETE'])
    n_pruned = len([t for t in study.trials if t.state.name == 'PRUNED'])
    print(f"\n📊 Trial Statistics:")
    print(f"  Complete: {n_complete}")
    print(f"  Pruned: {n_pruned}")
    print(f"  Pruning rate: {n_pruned / len(study.trials):.1%}")
    
    return study


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization from config file"
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML config file'
    )
    args = parser.parse_args()
    
    # Load config
    print(f"📄 Loading config: {args.config}")
    config = load_config(args.config)
    
    # Prepare data
    print(f"\n📊 Preparing data...")
    df = prepare_data(config)
    
    # Run optimization
    study = run_optimization(config, df)
    
    print(f"\n✅ Done! MLflow experiment: {config['experiment']['name']}")
    print(f"   View results: mlflow ui")


if __name__ == '__main__':
    main()