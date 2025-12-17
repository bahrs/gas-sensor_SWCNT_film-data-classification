"""
optuna_objectives.py

Optuna objective functions with MLflow integration for hyperparameter optimization.
Includes per-epoch pruning for faster optimization.
"""

import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

# Suppress other warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import numpy as np
from typing import Dict, Any, Optional
import logging

from models.lstm_model import build_lstm, train_lstm
from models.catboost_model import (
    build_catboost_classifier, train_catboost_classifier,
    build_catboost_regressor, train_catboost_regressor
)
from preprocessing.train_test import create_time_series_folds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LSTM OBJECTIVES
# ============================================================================

class LSTMRegressorObjective:
    """
    Optuna objective for LSTM regression hyperparameter optimization.
    
    Uses time-series CV to evaluate hyperparameters across multiple folds.
    Includes per-epoch pruning for faster optimization.
    """
    
    def __init__(
        self,
        df,
        look_back_range: tuple = (20, 60),
        n_components_range: tuple = (50, 150),
        cv_start_cycle: int = 7,
        cv_test_size: int = 1,
        feature_cols: int = 402,
        target_cols: list = None
    ):
        self.df = df
        self.look_back_range = look_back_range
        self.n_components_range = n_components_range
        self.cv_start_cycle = cv_start_cycle
        self.cv_test_size = cv_test_size
        self.feature_cols = feature_cols
        self.target_cols = target_cols or ['NO2', 'H2S', 'Acet']
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Returns:
            Mean validation RMSE across all CV folds (lower is better)
        """
        # Hyperparameters to optimize
        params = {
            'look_back': trial.suggest_int('look_back', *self.look_back_range),
            'n_components': trial.suggest_int('n_components', *self.n_components_range),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
            'n_units': trial.suggest_int('n_units', 32, 128, step=16),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'epochs': 150,
            'patience': 30
        }
        
        try:
            # Create CV splits with these hyperparameters
            cv_splitter = create_time_series_folds(
                self.df,
                model_type='lstm',
                task_type='regressor',
                feature_cols=self.feature_cols,
                target_cols=self.target_cols,
                look_back=params['look_back'],
                n_components=params['n_components'],
                start_cycle=self.cv_start_cycle,
                test_size=self.cv_test_size
            )
            
            # Train on each fold
            fold_rmses = []
            fold_gaps = []
            fold_epochs = []
            
            for fold_idx, fold in enumerate(cv_splitter):
                # Build model
                input_shape = (fold.train_X.shape[1], fold.train_X.shape[2])
                output_shape = fold.train_y.shape[1]
                
                model = build_lstm(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    n_layers=params['n_layers'],
                    n_units=params['n_units'],
                    dropout=params['dropout'],
                    learning_rate=params['learning_rate']
                )
                
                # Train model with pruning callback
                results = train_lstm(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    patience=params['patience'],
                    verbose=0,
                    trial=trial,  # Enable per-epoch pruning
                    fold_idx=fold_idx
                )
                
                # Check if pruned during training
                if results.get('pruned', False):
                    raise optuna.TrialPruned()
                
                fold_rmses.append(results['val_rmse'])
                fold_gaps.append(results['train_val_gap'])
                fold_epochs.append(results['best_epoch'])
                
                # Report after each fold for inter-fold pruning
                trial.report(results['val_rmse'], fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Aggregate metrics across folds
            mean_rmse = np.mean(fold_rmses)
            std_rmse = np.std(fold_rmses)
            mean_gap = np.mean(fold_gaps)
            mean_epochs = np.mean(fold_epochs)
            
            # Store metrics as user attributes (Optuna handles these properly)
            trial.set_user_attr('mean_rmse', float(mean_rmse))
            trial.set_user_attr('std_rmse', float(std_rmse))
            trial.set_user_attr('cv_stability', float(std_rmse / mean_rmse))
            trial.set_user_attr('mean_train_val_gap', float(mean_gap))
            trial.set_user_attr('mean_epochs', float(mean_epochs))
            
            # Store per-fold metrics
            for fold_idx, (rmse, gap, epochs) in enumerate(zip(fold_rmses, fold_gaps, fold_epochs)):
                trial.set_user_attr(f'fold_{fold_idx}_rmse', float(rmse))
                trial.set_user_attr(f'fold_{fold_idx}_gap', float(gap))
                trial.set_user_attr(f'fold_{fold_idx}_epochs', int(epochs))
            
            return mean_rmse
            
        except optuna.TrialPruned:
            # Re-raise pruned exception
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            trial.set_user_attr('error', str(e))
            raise


class LSTMClassifierObjective:
    """
    Optuna objective for LSTM classification hyperparameter optimization.
    """
    
    def __init__(
        self,
        df,
        look_back_range: tuple = (20, 60),
        n_components_range: tuple = (50, 150),
        cv_start_cycle: int = 7,
        cv_test_size: int = 1,
        feature_cols: int = 402
    ):
        self.df = df
        self.look_back_range = look_back_range
        self.n_components_range = n_components_range
        self.cv_start_cycle = cv_start_cycle
        self.cv_test_size = cv_test_size
        self.feature_cols = feature_cols
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Returns:
            Mean validation categorical accuracy across CV folds (higher is better)
        """
        params = {
            'look_back': trial.suggest_int('look_back', *self.look_back_range),
            'n_components': trial.suggest_int('n_components', *self.n_components_range),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
            'n_units': trial.suggest_int('n_units', 32, 128, step=16),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'epochs': 150,
            'patience': 30
        }
        
        try:
            cv_splitter = create_time_series_folds(
                self.df,
                model_type='lstm',
                task_type='classifier',
                feature_cols=self.feature_cols,
                look_back=params['look_back'],
                n_components=params['n_components'],
                start_cycle=self.cv_start_cycle,
                test_size=self.cv_test_size
            )
            
            fold_accuracies = []
            
            for fold_idx, fold in enumerate(cv_splitter):
                input_shape = (fold.train_X.shape[1], fold.train_X.shape[2])
                output_shape = fold.train_y.shape[1]  # Number of classes (one-hot)
                
                model = build_lstm(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    n_layers=params['n_layers'],
                    n_units=params['n_units'],
                    dropout=params['dropout'],
                    learning_rate=params['learning_rate'],
                    loss='categorical_crossentropy'
                )
                
                results = train_lstm(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    patience=params['patience'],
                    verbose=0,
                    trial=trial,
                    fold_idx=fold_idx
                )
                
                if results.get('pruned', False):
                    raise optuna.TrialPruned()
                
                # Get accuracy from final validation
                y_pred = model.predict(fold.test_X, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(fold.test_y, axis=1)
                accuracy = np.mean(y_pred_classes == y_true_classes)
                
                fold_accuracies.append(accuracy)
                
                trial.report(accuracy, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            
            trial.set_user_attr('mean_accuracy', float(mean_accuracy))
            trial.set_user_attr('std_accuracy', float(std_accuracy))
            
            for fold_idx, acc in enumerate(fold_accuracies):
                trial.set_user_attr(f'fold_{fold_idx}_accuracy', float(acc))
            
            return mean_accuracy
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            trial.set_user_attr('error', str(e))
            raise


# ============================================================================
# CATBOOST OBJECTIVES
# ============================================================================

class CatBoostClassifierObjective:
    """
    Optuna objective for CatBoost classification.
    """
    
    def __init__(
        self,
        df,
        n_components_range: tuple = (50, 200),
        cv_start_cycle: int = 7,
        cv_test_size: int = 1,
        feature_cols: int = 402
    ):
        self.df = df
        self.n_components_range = n_components_range
        self.cv_start_cycle = cv_start_cycle
        self.cv_test_size = cv_test_size
        self.feature_cols = feature_cols
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Returns:
            Mean F1-macro score across CV folds (higher is better)
        """
        params = {
            'n_components': trial.suggest_int('n_components', *self.n_components_range),
            'iterations': trial.suggest_int('iterations', 500, 2000, step=100),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0)
        }
        
        try:
            # Create CV splits
            cv_splitter = create_time_series_folds(
                self.df,
                model_type='catboost',
                task_type='classifier',
                feature_cols=self.feature_cols,
                n_components=params['n_components'],
                start_cycle=self.cv_start_cycle,
                test_size=self.cv_test_size
            )
            
            # Train on each fold
            fold_f1_scores = []
            fold_accuracies = []
            
            for fold_idx, fold in enumerate(cv_splitter):
                model = build_catboost_classifier(
                    iterations=params['iterations'],
                    depth=params['depth'],
                    learning_rate=params['learning_rate'],
                    l2_leaf_reg=params['l2_leaf_reg'],
                    verbose=False
                )
                
                metrics = train_catboost_classifier(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    early_stopping_rounds=50,
                    verbose=False,
                    trial=trial,  # Enable pruning
                    fold_idx=fold_idx
                )
                
                # Check if pruned
                if metrics.get('pruned', False):
                    raise optuna.TrialPruned()
                
                fold_f1_scores.append(metrics['val_f1_macro'])
                fold_accuracies.append(metrics['val_accuracy'])
                
                # Report after each fold
                trial.report(metrics['val_f1_macro'], fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Aggregate metrics
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)
            mean_acc = np.mean(fold_accuracies)
            
            trial.set_user_attr('mean_f1', float(mean_f1))
            trial.set_user_attr('std_f1', float(std_f1))
            trial.set_user_attr('mean_accuracy', float(mean_acc))
            
            for fold_idx, (f1, acc) in enumerate(zip(fold_f1_scores, fold_accuracies)):
                trial.set_user_attr(f'fold_{fold_idx}_f1', float(f1))
                trial.set_user_attr(f'fold_{fold_idx}_accuracy', float(acc))
            
            return mean_f1
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            trial.set_user_attr('error', str(e))
            raise


class CatBoostRegressorObjective:
    """
    Optuna objective for CatBoost regression.
    """
    
    def __init__(
        self,
        df,
        n_components_range: tuple = (50, 200),
        cv_start_cycle: int = 7,
        cv_test_size: int = 1,
        feature_cols: int = 402,
        target_cols: list = None
    ):
        self.df = df
        self.n_components_range = n_components_range
        self.cv_start_cycle = cv_start_cycle
        self.cv_test_size = cv_test_size
        self.feature_cols = feature_cols
        self.target_cols = target_cols or ['NO2', 'H2S', 'Acet']
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Returns:
            Mean validation RMSE across CV folds (lower is better)
        """
        params = {
            'n_components': trial.suggest_int('n_components', *self.n_components_range),
            'iterations': trial.suggest_int('iterations', 500, 2000, step=100),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0)
        }
        
        try:
            cv_splitter = create_time_series_folds(
                self.df,
                model_type='catboost',
                task_type='regressor',
                feature_cols=self.feature_cols,
                target_cols=self.target_cols,
                n_components=params['n_components'],
                start_cycle=self.cv_start_cycle,
                test_size=self.cv_test_size
            )
            
            fold_rmses = []
            
            for fold_idx, fold in enumerate(cv_splitter):
                model = build_catboost_regressor(
                    iterations=params['iterations'],
                    depth=params['depth'],
                    learning_rate=params['learning_rate'],
                    l2_leaf_reg=params['l2_leaf_reg'],
                    verbose=False
                )
                
                metrics = train_catboost_regressor(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    early_stopping_rounds=50,
                    verbose=False,
                    trial=trial,
                    fold_idx=fold_idx
                )
                
                if metrics.get('pruned', False):
                    raise optuna.TrialPruned()
                
                fold_rmses.append(metrics['val_rmse'])
                
                trial.report(metrics['val_rmse'], fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            mean_rmse = np.mean(fold_rmses)
            std_rmse = np.std(fold_rmses)
            
            trial.set_user_attr('mean_rmse', float(mean_rmse))
            trial.set_user_attr('std_rmse', float(std_rmse))
            trial.set_user_attr('cv_stability', float(std_rmse / mean_rmse))
            
            for fold_idx, rmse in enumerate(fold_rmses):
                trial.set_user_attr(f'fold_{fold_idx}_rmse', float(rmse))
            
            return mean_rmse
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            trial.set_user_attr('error', str(e))
            raise


# ============================================================================
# OPTIMIZATION RUNNERS
# ============================================================================

def run_optimization(
    objective_class,
    df,
    study_name: str,
    mlflow_experiment_name: str,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    direction: str = "minimize",
    objective_kwargs: Optional[Dict] = None,
    storage: Optional[str] = None,
    pruner_config: Optional[Dict] = None
) -> optuna.Study:
    """
    Args:
        objective_class: Objective class (e.g., LSTMRegressorObjective)
        df: Preprocessed dataframe
        study_name: Optuna study name
        mlflow_experiment_name: MLflow experiment name
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds (None = no timeout)
        direction: 'minimize' or 'maximize'
        objective_kwargs: Additional kwargs for objective __init__
        storage: Optuna storage backend (default: SQLite)
        pruner_config: Pruner configuration dict
    
    Returns:
        Completed Optuna study
    """
    # Use SQLite storage by default for persistence
    if storage is None:
        storage = "sqlite:///optuna_studies.db"
    
    # Set MLflow to use SQLite backend
    mlflow_tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    # Defensive: close any previously active run
    while mlflow.active_run() is not None:
        mlflow.end_run()

    # Configure pruner
    if pruner_config is None:
        pruner_config = {
            'n_startup_trials': 5,
            'n_warmup_steps': 1,
            'interval_steps': 1
        }
    
    pruner = optuna.pruners.MedianPruner(**pruner_config)

    # Create Optuna study with persistent storage
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=True,  # Resume if study exists
        pruner=pruner
    )

    # Create objective
    objective_kwargs = objective_kwargs or {}
    objective = objective_class(df, **objective_kwargs)

    logger.info(f"Starting optimization: {study_name}")
    logger.info(f"Storage: {storage}")
    logger.info(f"MLflow: {mlflow_tracking_uri}")

    # Parent run for the whole study; trials will be nested runs
    with mlflow.start_run(run_name=f"study_{study_name}") as parent_run:
        # Create MLflow callback (nested trial runs)
        mlflc = MLflowCallback(
            tracking_uri=mlflow_tracking_uri,
            metric_name="value",
            mlflow_kwargs={"nested": True}
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflc],
            show_progress_bar=True,
            n_jobs=1  # Don't parallelize (MLflow logging issues)
        )

        # Log summary to parent run
        mlflow.log_params({f"best__{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "pruning_rate": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]) / len(study.trials)
        })

        # Link to best trial run
        try:
            best_trial = study.best_trial
            run_id = best_trial.system_attrs.get("mlflow_run_id") or best_trial.system_attrs.get("run_id")
            if run_id:
                mlflow.set_tag("best_trial_run_id", run_id)
        except Exception:
            pass

    logger.info(f"Optimization complete. Best value: {study.best_value}")
    logger.info(f"Pruning rate: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]) / len(study.trials):.1%}")
    
    return study


# Convenience functions for each model type

def run_lstm_regressor_optimization(df, **kwargs) -> optuna.Study:
    """Run LSTM regression optimization."""
    defaults = {
        'study_name': 'lstm_regression',
        'mlflow_experiment_name': 'SWCNT_LSTM_Regression',
        'n_trials': 100,
        'direction': 'minimize'
    }
    defaults.update(kwargs)
    return run_optimization(LSTMRegressorObjective, df, **defaults)


def run_lstm_classifier_optimization(df, **kwargs) -> optuna.Study:
    """Run LSTM classification optimization."""
    defaults = {
        'study_name': 'lstm_classification',
        'mlflow_experiment_name': 'SWCNT_LSTM_Classification',
        'n_trials': 100,
        'direction': 'maximize'
    }
    defaults.update(kwargs)
    return run_optimization(LSTMClassifierObjective, df, **defaults)


def run_catboost_classifier_optimization(df, **kwargs) -> optuna.Study:
    """Run CatBoost classification optimization."""
    defaults = {
        'study_name': 'catboost_classification',
        'mlflow_experiment_name': 'SWCNT_CatBoost_Classification',
        'n_trials': 100,
        'direction': 'maximize'
    }
    defaults.update(kwargs)
    return run_optimization(CatBoostClassifierObjective, df, **defaults)


def run_catboost_regressor_optimization(df, **kwargs) -> optuna.Study:
    """Run CatBoost regression optimization."""
    defaults = {
        'study_name': 'catboost_regression',
        'mlflow_experiment_name': 'SWCNT_CatBoost_Regression',
        'n_trials': 100,
        'direction': 'minimize'
    }
    defaults.update(kwargs)
    return run_optimization(CatBoostRegressorObjective, df, **defaults)