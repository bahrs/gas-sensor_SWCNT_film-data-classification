"""
cli.py

Command-line interface for SWCNT gas sensor ML pipeline.

Usage:
  thermocycling optimize configs/config_lstm_regression.yaml
  thermocycling optimize configs/config_catboost_regression.yaml --mlflow-uri sqlite:///custom.db
"""

import typer
import yaml
from pathlib import Path
from typing import Optional

from thermocycling.config import RunConfig
from thermocycling.utils.seed import set_global_seed
from thermocycling.pipeline.assemble import full_dataset
from thermocycling.preprocessing.smoothing import Exp_pd, Savitzky_Golay
from thermocycling.models.optuna_objectives import (
    run_lstm_regressor_optimization,
    run_lstm_classifier_optimization,
    run_catboost_regressor_optimization,
    run_catboost_classifier_optimization,
)

app = typer.Typer(
    name="thermocycling",
    help="🌡️ SWCNT Gas Sensor ML Pipeline - Hyperparameter optimization and training",
    add_completion=False,
)

DEDRIFT_METHODS = {
    "Exp_pd": Exp_pd,
    "Savitzky_Golay": Savitzky_Golay,
}


@app.command()
def optimize(
    config: str = typer.Argument(..., help="Path to YAML config file"),
    mlflow_uri: Optional[str] = typer.Option(None, "--mlflow-uri", help="Override MLflow tracking URI"),
    optuna_storage: Optional[str] = typer.Option(None, "--optuna-storage", help="Override Optuna storage URI"),
    n_trials: Optional[int] = typer.Option(None, "--n-trials", "-n", help="Override number of trials"),
):
    """Run hyperparameter optimization from config file."""
    
    # Load config
    cfg_path = Path(config).resolve()
    if not cfg_path.exists():
        typer.secho(f"❌ Config file not found: {cfg_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    
    typer.echo(f"📄 Loading config: {cfg_path}")
    raw_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    
    # Validate config
    try:
        cfg = RunConfig.model_validate(raw_config)
    except Exception as e:
        typer.secho(f"❌ Config validation failed:", fg=typer.colors.RED, err=True)
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    
    # Apply CLI overrides
    if mlflow_uri:
        cfg.mlflow.tracking_uri = mlflow_uri
    if optuna_storage:
        cfg.storage.optuna = optuna_storage
    if n_trials:
        cfg.optimization.n_trials = n_trials
    
    # Set seed
    set_global_seed(cfg.random_seed)
    typer.echo(f"🎲 Random seed: {cfg.random_seed}")
    
    # Load data
    typer.echo(f"\n📊 Preparing data...")
    dedrift_func = DEDRIFT_METHODS[cfg.data.dedrift_method]
    df = full_dataset(dedrifting_func=dedrift_func, **cfg.data.dedrift_params)
    
    typer.secho(f"✅ Loaded dataset: {df.shape}", fg=typer.colors.GREEN)
    typer.echo(f"   Gases: {list(df['gas_'].unique())}")
    typer.echo(f"   Cycles: {df['meas_cycle'].min()} → {df['meas_cycle'].max()}")
    typer.echo(f"   Targets: {cfg.data.target_cols}")
    
    # Build objective kwargs
    objective_kwargs = {
        "model_config": cfg.model,
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
    optimize_func = func_map.get(key)
    
    if not optimize_func:
        typer.secho(f"❌ Unknown model_type/task: {key}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    
    # Display optimization config
    typer.echo(f"\n🚀 Starting optimization:")
    typer.echo(f"   Experiment: {cfg.experiment.name}")
    typer.echo(f"   Study: {cfg.experiment.study_name}")
    typer.echo(f"   Model: {cfg.experiment.model_type} ({cfg.experiment.task})")
    typer.echo(f"   Trials: {cfg.optimization.n_trials}")
    typer.echo(f"   Direction: {cfg.optimization.direction}")
    typer.echo(f"   Optuna: {cfg.storage.optuna}")
    typer.echo(f"   MLflow: {cfg.mlflow.tracking_uri}")
    typer.echo(f"   Results: {cfg.results_dir}/")
    
    # Run optimization
    try:
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
            config_dump=raw_config,
        )
        
        typer.secho(f"\n✅ Optimization complete!", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"   Best value: {study.best_value:.6f}")
        typer.echo(f"   Best params: {study.best_params}")
        typer.echo(f"\n📊 View results:")
        typer.echo(f"   Files: {cfg.results_dir}/")
        typer.echo(f"   MLflow UI: mlflow ui --backend-store-uri {cfg.mlflow.tracking_uri}")
        
    except KeyboardInterrupt:
        typer.secho("\n⚠️  Optimization interrupted by user", fg=typer.colors.YELLOW)
        raise typer.Exit(130)
    except Exception as e:
        typer.secho(f"\n❌ Optimization failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    config: str = typer.Argument(..., help="Path to YAML config file"),
):
    """Validate a config file without running optimization."""
    
    cfg_path = Path(config).resolve()
    if not cfg_path.exists():
        typer.secho(f"❌ Config file not found: {cfg_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    
    typer.echo(f"📄 Validating config: {cfg_path}")
    raw_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    
    try:
        cfg = RunConfig.model_validate(raw_config)
        typer.secho("✅ Config is valid!", fg=typer.colors.GREEN, bold=True)
        
        # Display summary
        typer.echo(f"\n📋 Configuration summary:")
        typer.echo(f"   Experiment: {cfg.experiment.name}")
        typer.echo(f"   Model: {cfg.experiment.model_type} ({cfg.experiment.task})")
        typer.echo(f"   Targets: {cfg.data.target_cols}")
        typer.echo(f"   CV: {cfg.preprocessing.cv_start_cycle}+ cycles, test_size={cfg.preprocessing.cv_test_size}")
        typer.echo(f"   Trials: {cfg.optimization.n_trials}")
        typer.echo(f"   Seed: {cfg.random_seed}")
        
    except Exception as e:
        typer.secho(f"❌ Config validation failed:", fg=typer.colors.RED, err=True)
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def info():
    """Display package information and available commands."""
    
    typer.secho("🌡️  SWCNT Gas Sensor ML Pipeline", fg=typer.colors.CYAN, bold=True)
    typer.echo("\nA production-ready ML pipeline for gas classification and concentration")
    typer.echo("prediction from thermocycled SWCNT sensor data.")
    typer.echo("\nAvailable commands:")
    typer.echo("  optimize    Run hyperparameter optimization")
    typer.echo("  validate    Validate configuration file")
    typer.echo("  info        Show this information")
    typer.echo("\nExamples:")
    typer.echo("  thermocycling optimize configs/config_lstm_regression.yaml")
    typer.echo("  thermocycling optimize configs/config_catboost.yaml --n-trials 50")
    typer.echo("  thermocycling validate configs/config_lstm_regression.yaml")
    typer.echo("\nDocumentation: https://github.com/bahrs/gas-sensor_SWCNT_film-data-classification")


if __name__ == "__main__":
    app()