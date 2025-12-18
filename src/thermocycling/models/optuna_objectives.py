"""
optuna_objectives.py

Optuna objective functions with MLflow integration for hyperparameter optimization.
- All hyperparameter ranges come from YAML via model_config dict
- MLflow tracking URI and Optuna storage are fully configurable
- Auto-exports results to ./results with full config and trials data
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Sequence

import warnings
import numpy as np
import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback

# Reduce noisy warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _get_range(config: Dict, key: str, default: Sequence) -> tuple:
    """Extract a 2-element range from config, with fallback."""
    val = config.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return tuple(val)
    raise ValueError(f"Expected 2-element range for {key}, got: {val!r}")


def export_study_results(
    study: optuna.Study,
    out_dir: Path,
    config_dump: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write study artifacts to disk:
    - best_params.yaml
    - summary.json (includes full config used)
    - trials.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best params as YAML
    try:
        import yaml
        (out_dir / "best_params.yaml").write_text(
            yaml.safe_dump(study.best_params, sort_keys=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"Failed to write best_params.yaml: {e}")

    # Summary JSON
    summary = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials_total": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_used": config_dump,  # Full YAML config for reproducibility
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Trials dataframe as CSV
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(out_dir / "trials.csv", index=False)
    except Exception as e:
        logger.warning(f"Failed to write trials.csv: {e}")

    logger.info(f"✅ Results exported to {out_dir}")
    return out_dir


# =============================================================================
# Base Objective Class (Shared Logic)
# =============================================================================

class BaseObjective:
    """
    Base class for all objectives with common config extraction logic.
    Subclasses only need to implement __call__().
    """
    
    def __init__(
        self,
        df,
        model_config: Dict[str, Any],
        cv_start_cycle: int = 7,
        cv_test_size: int = 1,
        feature_cols: int = 402,
        target_cols: Optional[Sequence[str]] = None,
        random_seed: int = 42,
    ):
        self.df = df
        self.config = model_config
        self.cv_start_cycle = int(cv_start_cycle)
        self.cv_test_size = int(cv_test_size)
        self.feature_cols = int(feature_cols)
        self.target_cols = list(target_cols) if target_cols else []
        self.random_seed = int(random_seed)


# =============================================================================
# LSTM Objectives
# =============================================================================

class LSTMRegressorObjective(BaseObjective):
    """Optuna objective for LSTM regression. Minimizes mean CV RMSE."""

    def __call__(self, trial: optuna.Trial) -> float:
        from thermocycling.models.lstm_model import build_lstm, train_lstm
        from thermocycling.preprocessing.train_test import create_time_series_folds
        
        # Extract hyperparameters from config with trial suggestions
        params = {
            "look_back": trial.suggest_int("look_back", *_get_range(self.config, "look_back_range", [20, 60])),
            "n_components": trial.suggest_int("n_components", *_get_range(self.config, "n_components_range", [10, 100])),
            "n_layers": trial.suggest_int("n_layers", *_get_range(self.config, "n_layers_range", [1, 3])),
            "n_units": trial.suggest_int(
                "n_units", 
                *_get_range(self.config, "n_units_range", [32, 128]),
                step=self.config.get("n_units_step", 16)
            ),
            "dropout": trial.suggest_float("dropout", *_get_range(self.config, "dropout_range", [0.1, 0.5])),
            "learning_rate": trial.suggest_float(
                "learning_rate", 
                *_get_range(self.config, "learning_rate_range", [1e-4, 1e-2]),
                log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", self.config.get("batch_size_options", [32, 64, 128])),
            "epochs": self.config.get("epochs", 100),
            "patience": self.config.get("patience", 20),
        }

        cv_splitter = create_time_series_folds(
            self.df,
            model_type="lstm",
            task_type="regressor",
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            look_back=params["look_back"],
            n_components=params["n_components"],
            start_cycle=self.cv_start_cycle,
            test_size=self.cv_test_size,
        )

        fold_rmses = []
        fold_gaps = []

        try:
            for fold_idx, fold in enumerate(cv_splitter.folds):
                model = build_lstm(
                    input_shape=(fold.train_X.shape[1], fold.train_X.shape[2]),
                    output_shape=fold.train_y.shape[1],
                    n_layers=params["n_layers"],
                    n_units=params["n_units"],
                    dropout=params["dropout"],
                    learning_rate=params["learning_rate"],
                    loss="mse",
                )

                results = train_lstm(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    patience=params["patience"],
                    verbose=0,
                    trial=trial,
                    fold_idx=fold_idx,
                )

                if results.get("pruned", False):
                    raise optuna.TrialPruned()

                fold_rmses.append(float(results["val_rmse"]))
                fold_gaps.append(float(results.get("train_val_gap", 0.0)))

                trial.report(float(np.mean(fold_rmses)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_rmse = float(np.mean(fold_rmses))
            std_rmse = float(np.std(fold_rmses))

            trial.set_user_attr("mean_rmse", mean_rmse)
            trial.set_user_attr("std_rmse", std_rmse)
            trial.set_user_attr("cv_stability", std_rmse / mean_rmse if mean_rmse > 0 else 0.0)
            trial.set_user_attr("mean_train_val_gap", float(np.mean(fold_gaps)))

            return mean_rmse

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.exception(f"Trial {trial.number} failed")
            trial.set_user_attr("error", str(e))
            raise


class LSTMClassifierObjective(BaseObjective):
    """Optuna objective for LSTM classification. Maximizes mean CV accuracy."""

    def __call__(self, trial: optuna.Trial) -> float:
        from thermocycling.models.lstm_model import build_lstm, train_lstm
        from thermocycling.preprocessing.train_test import create_time_series_folds
        
        params = {
            "look_back": trial.suggest_int("look_back", *_get_range(self.config, "look_back_range", [20, 60])),
            "n_components": trial.suggest_int("n_components", *_get_range(self.config, "n_components_range", [10, 100])),
            "n_layers": trial.suggest_int("n_layers", *_get_range(self.config, "n_layers_range", [1, 3])),
            "n_units": trial.suggest_int(
                "n_units", 
                *_get_range(self.config, "n_units_range", [32, 128]),
                step=self.config.get("n_units_step", 16)
            ),
            "dropout": trial.suggest_float("dropout", *_get_range(self.config, "dropout_range", [0.1, 0.5])),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                *_get_range(self.config, "learning_rate_range", [1e-4, 1e-2]),
                log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", self.config.get("batch_size_options", [32, 64, 128])),
            "epochs": self.config.get("epochs", 100),
            "patience": self.config.get("patience", 20),
        }

        cv_splitter = create_time_series_folds(
            self.df,
            model_type="lstm",
            task_type="classifier",
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            look_back=params["look_back"],
            n_components=params["n_components"],
            start_cycle=self.cv_start_cycle,
            test_size=self.cv_test_size,
        )

        fold_accs = []

        try:
            for fold_idx, fold in enumerate(cv_splitter.folds):
                model = build_lstm(
                    input_shape=(fold.train_X.shape[1], fold.train_X.shape[2]),
                    output_shape=fold.train_y.shape[1],
                    n_layers=params["n_layers"],
                    n_units=params["n_units"],
                    dropout=params["dropout"],
                    learning_rate=params["learning_rate"],
                    loss="categorical_crossentropy",
                )

                results = train_lstm(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    patience=params["patience"],
                    verbose=0,
                    trial=trial,
                    fold_idx=fold_idx,
                )

                if results.get("pruned", False):
                    raise optuna.TrialPruned()

                y_pred = model.predict(fold.test_X, verbose=0)
                acc = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(fold.test_y, axis=1)))
                fold_accs.append(acc)

                trial.report(float(np.mean(fold_accs)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_acc = float(np.mean(fold_accs))
            trial.set_user_attr("mean_accuracy", mean_acc)
            trial.set_user_attr("std_accuracy", float(np.std(fold_accs)))

            return mean_acc

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.exception(f"Trial {trial.number} failed")
            trial.set_user_attr("error", str(e))
            raise


# =============================================================================
# CatBoost Objectives
# =============================================================================

class CatBoostRegressorObjective(BaseObjective):
    """Optuna objective for CatBoost regression. Minimizes mean CV RMSE."""

    def __call__(self, trial: optuna.Trial) -> float:
        from thermocycling.models.catboost_model import build_catboost_regressor, train_catboost_regressor
        from thermocycling.preprocessing.train_test import create_time_series_folds
        
        params = {
            "n_components": trial.suggest_int("n_components", *_get_range(self.config, "n_components_range", [50, 200])),
            "iterations": trial.suggest_int(
                "iterations",
                *_get_range(self.config, "iterations_range", [400, 2000]),
                step=self.config.get("iterations_step", 100)
            ),
            "depth": trial.suggest_int("depth", *_get_range(self.config, "depth_range", [4, 8])),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                *_get_range(self.config, "learning_rate_range", [0.001, 0.3]),
                log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *_get_range(self.config, "l2_leaf_reg_range", [3.0, 20.0])),
        }

        cv_splitter = create_time_series_folds(
            self.df,
            model_type="catboost",
            task_type="regressor",
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            n_components=params["n_components"],
            start_cycle=self.cv_start_cycle,
            test_size=self.cv_test_size,
        )

        fold_rmses = []

        try:
            for fold_idx, fold in enumerate(cv_splitter.folds):
                model = build_catboost_regressor(
                    iterations=params["iterations"],
                    depth=params["depth"],
                    learning_rate=params["learning_rate"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                    loss_function=self.config.get("loss_function", "MultiRMSE"),
                    verbose=False,
                    random_seed=self.random_seed,
                )

                metrics = train_catboost_regressor(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
                    verbose=False,
                    trial=trial,
                    fold_idx=fold_idx,
                )

                if metrics.get("pruned", False):
                    raise optuna.TrialPruned()

                fold_rmses.append(float(metrics["val_rmse"]))

                trial.report(float(np.mean(fold_rmses)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_rmse = float(np.mean(fold_rmses))
            trial.set_user_attr("mean_rmse", mean_rmse)
            trial.set_user_attr("std_rmse", float(np.std(fold_rmses)))
            trial.set_user_attr("cv_stability", float(np.std(fold_rmses) / mean_rmse) if mean_rmse > 0 else 0.0)

            return mean_rmse

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.exception(f"Trial {trial.number} failed")
            trial.set_user_attr("error", str(e))
            raise


class CatBoostClassifierObjective(BaseObjective):
    """Optuna objective for CatBoost classification. Maximizes mean CV F1-macro."""

    def __call__(self, trial: optuna.Trial) -> float:
        from thermocycling.models.catboost_model import build_catboost_classifier, train_catboost_classifier
        from thermocycling.preprocessing.train_test import create_time_series_folds
        
        params = {
            "n_components": trial.suggest_int("n_components", *_get_range(self.config, "n_components_range", [50, 200])),
            "iterations": trial.suggest_int(
                "iterations",
                *_get_range(self.config, "iterations_range", [400, 2000]),
                step=self.config.get("iterations_step", 100)
            ),
            "depth": trial.suggest_int("depth", *_get_range(self.config, "depth_range", [4, 8])),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                *_get_range(self.config, "learning_rate_range", [0.001, 0.3]),
                log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *_get_range(self.config, "l2_leaf_reg_range", [3.0, 20.0])),
        }

        cv_splitter = create_time_series_folds(
            self.df,
            model_type="catboost",
            task_type="classifier",
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            n_components=params["n_components"],
            start_cycle=self.cv_start_cycle,
            test_size=self.cv_test_size,
        )

        fold_f1 = []

        try:
            for fold_idx, fold in enumerate(cv_splitter.folds):
                model = build_catboost_classifier(
                    iterations=params["iterations"],
                    depth=params["depth"],
                    learning_rate=params["learning_rate"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                    loss_function=self.config.get("loss_function", "MultiClass"),
                    eval_metric=self.config.get("eval_metric", "TotalF1"),
                    verbose=False,
                    random_seed=self.random_seed,
                )

                metrics = train_catboost_classifier(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
                    verbose=False,
                    trial=trial,
                    fold_idx=fold_idx,
                )

                if metrics.get("pruned", False):
                    raise optuna.TrialPruned()

                fold_f1.append(float(metrics["val_f1_macro"]))

                trial.report(float(np.mean(fold_f1)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_f1 = float(np.mean(fold_f1))
            trial.set_user_attr("mean_f1_macro", mean_f1)
            trial.set_user_attr("std_f1_macro", float(np.std(fold_f1)))

            return mean_f1

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.exception(f"Trial {trial.number} failed")
            trial.set_user_attr("error", str(e))
            raise


# =============================================================================
# Main Optimization Runner
# =============================================================================

def run_optimization(
    objective_class,
    df,
    study_name: str,
    mlflow_experiment_name: str,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    direction: str = "minimize",
    objective_kwargs: Optional[Dict[str, Any]] = None,
    storage: Optional[str] = None,
    pruner_config: Optional[Dict[str, Any]] = None,
    mlflow_tracking_uri: Optional[str] = None,
    results_dir: str = "results",
    config_dump: Optional[Dict[str, Any]] = None,
) -> optuna.Study:
    """
    Run Optuna study with MLflow logging and auto-export results.
    
    Args:
        objective_class: Objective class to instantiate
        df: Preprocessed dataframe
        study_name: Optuna study name
        mlflow_experiment_name: MLflow experiment name
        n_trials: Number of optimization trials
        timeout: Timeout in seconds (None = no limit)
        direction: "minimize" or "maximize"
        objective_kwargs: Kwargs for objective class (includes model_config)
        storage: Optuna storage URI (defaults to env var or sqlite)
        pruner_config: Pruner configuration dict
        mlflow_tracking_uri: MLflow tracking URI (defaults to env var or sqlite)
        results_dir: Directory to save results
        config_dump: Full config dict to save with results
    
    Returns:
        Completed Optuna study
    """
    # Resolve storage URIs with env fallbacks
    if storage is None:
        storage = os.getenv("OPTUNA_STORAGE", "sqlite:///optuna_studies.db")
    
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    # Configure pruner
    if pruner_config is None:
        pruner_config = {"n_startup_trials": 5, "n_warmup_steps": 1, "interval_steps": 1}
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_config.get("n_startup_trials", 5),
        n_warmup_steps=pruner_config.get("n_warmup_steps", 1),
        interval_steps=pruner_config.get("interval_steps", 1)
    )

    # Close any stale MLflow runs
    while mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    logger.info(f"Optuna storage: {storage}")
    logger.info(f"MLflow tracking: {mlflow_tracking_uri}")
    logger.info(f"Study: {study_name} | {direction} | {n_trials} trials")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )

    # Run optimization with MLflow parent run
    with mlflow.start_run(run_name=f"study_{study_name}") as parent_run:
        mlflow.set_tag("study_name", study_name)
        mlflow.set_tag("objective", objective_class.__name__)

        # MLflow callback for per-trial nested runs
        mlflow_cb = MLflowCallback(
            tracking_uri=mlflow_tracking_uri,
            metric_name="objective_value",
            mlflow_kwargs={"nested": True},
        )

        objective = objective_class(df=df, **(objective_kwargs or {}))

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflow_cb],
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # Export results to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(results_dir) / f"{study_name}__{timestamp}"
        export_study_results(study, out_dir, config_dump=config_dump)

        # Log artifacts to MLflow parent run
        try:
            mlflow.log_artifacts(str(out_dir), artifact_path="results")
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

        # Log summary metrics
        mlflow.log_metric("best_value", float(study.best_value))
        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.log_metric("n_complete", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
        mlflow.log_metric("n_pruned", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))

        logger.info(f"✅ Optimization complete. Best value: {study.best_value:.6f}")

    return study


# =============================================================================
# Convenience Wrappers
# =============================================================================

def run_lstm_regressor_optimization(df, **kwargs) -> optuna.Study:
    defaults = {
        "study_name": "lstm_regression",
        "mlflow_experiment_name": "SWCNT_LSTM_Regression",
        "direction": "minimize",
    }
    defaults.update(kwargs)
    return run_optimization(LSTMRegressorObjective, df, **defaults)


def run_lstm_classifier_optimization(df, **kwargs) -> optuna.Study:
    defaults = {
        "study_name": "lstm_classification",
        "mlflow_experiment_name": "SWCNT_LSTM_Classification",
        "direction": "maximize",
    }
    defaults.update(kwargs)
    return run_optimization(LSTMClassifierObjective, df, **defaults)


def run_catboost_regressor_optimization(df, **kwargs) -> optuna.Study:
    defaults = {
        "study_name": "catboost_regression",
        "mlflow_experiment_name": "SWCNT_CatBoost_Regression",
        "direction": "minimize",
    }
    defaults.update(kwargs)
    return run_optimization(CatBoostRegressorObjective, df, **defaults)


def run_catboost_classifier_optimization(df, **kwargs) -> optuna.Study:
    defaults = {
        "study_name": "catboost_classification",
        "mlflow_experiment_name": "SWCNT_CatBoost_Classification",
        "direction": "maximize",
    }
    defaults.update(kwargs)
    return run_optimization(CatBoostClassifierObjective, df, **defaults)