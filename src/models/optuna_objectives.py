"""
optuna_objectives.py

Optuna objective functions with MLflow integration for hyperparameter optimization.
"""

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import numpy as np
from typing import Dict, Any, Optional
import logging

from src.models.lstm_model import build_lstm, train_lstm
from src.models.catboost_model import (
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
        
        # Log all params to MLflow
        mlflow.log_params(params)
        
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
                
                # Train model
                results = train_lstm(
                    model,
                    fold.train_X, fold.train_y,
                    fold.test_X, fold.test_y,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    patience=params['patience'],
                    verbose=0
                )
                
                fold_rmses.append(results['val_rmse'])
                fold_gaps.append(results['train_val_gap'])
                
                # Log fold-specific metrics
                mlflow.log_metrics({
                    f'fold_{fold_idx}_rmse': results['val_rmse'],
                    f'fold_{fold_idx}_train_val_gap': results['train_val_gap'],
                    f'fold_{fold_idx}_best_epoch': results['best_epoch']
                })
                
                # Prune trial if performance is poor
                trial.report(results['val_rmse'], fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Aggregate metrics across folds
            mean_rmse = np.mean(fold_rmses)
            std_rmse = np.std(fold_rmses)
            mean_gap = np.mean(fold_gaps)
            
            mlflow.log_metrics({
                'mean_cv_rmse': mean_rmse,
                'std_cv_rmse': std_rmse,
                'mean_train_val_gap': mean_gap,
                'n_folds': len(fold_rmses)
            })
            
            return mean_rmse
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            mlflow.log_param('error', str(e))
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
        
        mlflow.log_params(params)
        
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
                    verbose=0
                )
                
                # Get accuracy from final validation loss
                y_pred = model.predict(fold.test_X, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(fold.test_y, axis=1)
                accuracy = np.mean(y_pred_classes == y_true_classes)
                
                fold_accuracies.append(accuracy)
                
                mlflow.log_metrics({
                    f'fold_{fold_idx}_accuracy': accuracy,
                    f'fold_{fold_idx}_val_loss': results['final_val_loss']
                })
                
                trial.report(accuracy, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            
            mlflow.log_metrics({
                'mean_cv_accuracy': mean_accuracy,
                'std_cv_accuracy': std_accuracy,
                'n_folds': len(fold_accuracies)
            })
            
            return mean_accuracy
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            mlflow.log_param('error', str(e))
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
        
        mlflow.log_params(params)
        
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
                    verbose=False
                )
                
                fold_f1_scores.append(metrics['val_f1_macro'])
                fold_accuracies.append(metrics['val_accuracy'])
                
                # Log fold metrics
                mlflow.log_metrics({
                    f'fold_{fold_idx}_f1_macro': metrics['val_f1_macro'],
                    f'fold_{fold_idx}_accuracy': metrics['val_accuracy'],
                    f'fold_{fold_idx}_best_iteration': metrics['best_iteration']
                })
                
                # Prune if poor performance
                trial.report(metrics['val_f1_macro'], fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Aggregate metrics
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)
            mean_acc = np.mean(fold_accuracies)
            
            mlflow.log_metrics({
                'mean_cv_f1_macro': mean_f1,
                'std_cv_f1_macro': std_f1,
                'mean_cv_accuracy': mean_acc,
                'n_folds': len(fold_f1_scores)
            })
            
            return mean_f1
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            mlflow.log_param('error', str(e))
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
        
        mlflow.log_params(params)
        
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
                    verbose=False
                )
                
                fold_rmses.append(metrics['val_rmse'])
                
                mlflow.log_metrics({
                    f'fold_{fold_idx}_rmse': metrics['val_rmse'],
                    f'fold_{fold_idx}_mae': metrics['val_mae'],
                    f'fold_{fold_idx}_best_iteration': metrics['best_iteration']
                })
                
                trial.report(metrics['val_rmse'], fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            mean_rmse = np.mean(fold_rmses)
            std_rmse = np.std(fold_rmses)
            
            mlflow.log_metrics({
                'mean_cv_rmse': mean_rmse,
                'std_cv_rmse': std_rmse,
                'n_folds': len(fold_rmses)
            })
            
            return mean_rmse
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            mlflow.log_param('error', str(e))
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
    direction: str = 'minimize',
    objective_kwargs: Optional[Dict] = None
) -> optuna.Study:
    """
    Generic optimization runner with MLflow integration.
    
    Args:
        objective_class: Objective class (e.g., LSTMRegressorObjective)
        df: Preprocessed dataframe
        study_name: Optuna study name
        mlflow_experiment_name: MLflow experiment name
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds (None = no timeout)
        direction: 'minimize' or 'maximize'
        objective_kwargs: Additional kwargs for objective __init__
    
    Returns:
        Completed Optuna study
    """
    # Set MLflow experiment
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=2,
            interval_steps=1
        )
    )
    
    # Create MLflow callback
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name='value'
    )
    
    # Create objective
    objective_kwargs = objective_kwargs or {}
    objective = objective_class(df, **objective_kwargs)
    
    # Run optimization
    logger.info(f"Starting optimization: {study_name}")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[mlflc],
        show_progress_bar=True,
        n_jobs=1  # Don't parallelize (MLflow logging issues)
    )
    
    # Log best parameters to MLflow
    with mlflow.start_run(run_name=f'best_{study_name}'):
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        })
    
    logger.info(f"Optimization complete. Best value: {study.best_value}")
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