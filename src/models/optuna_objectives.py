"""
optuna_objectives.py

Optuna objective functions with MLflow integration for hyperparameter optimization.
"""

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import numpy as np
from typing import Dict, Any

from src.models.lstm_model import build_lstm, train_lstm, evaluate_lstm
from src.models.catboost_model import (
    build_catboost_classifier, train_catboost_classifier,
    build_catboost_regressor, train_catboost_regressor
)
from src.data.preprocessing import load_processed_data
from src.data.train_test_split import create_cv_splits_for_lstm, create_cv_splits_for_catboost


class LSTMObjective:
    """
    Optuna objective for LSTM hyperparameter optimization.
    
    Uses time-series CV to evaluate hyperparameters across multiple folds.
    """
    
    def __init__(
        self,
        df,
        look_back_range: tuple = (20, 60),
        n_components_range: tuple = (50, 150),
        cv_start_cycle: int = 7
    ):
        self.df = df
        self.look_back_range = look_back_range
        self.n_components_range = n_components_range
        self.cv_start_cycle = cv_start_cycle
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Returns:
            Mean validation RMSE across all CV folds
        """
        # Hyperparameters to optimize
        params = {
            'look_back': trial.suggest_int('look_back', *self.look_back_range),
            'n_components': trial.suggest_int('n_components', *self.n_components_range),
            'n_layers': trial.suggest_int('n_layers', 1, 2),
            'n_units': trial.suggest_int('n_units', 32, 128, step=16),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'epochs': 150,  # Fixed (early stopping will handle this)
            'patience': 30
        }
        
        # Log all params to MLflow
        mlflow.log_params(params)
        
        # Create CV splits with these hyperparameters
        cv_splitter = create_cv_splits_for_lstm(
            self.df,
            look_back=params['look_back'],
            n_components=params['n_components'],
            start_cycle=self.cv_start_cycle,
            do_pca=True
        )
        
        # Train on each fold
        fold_rmses = []
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
            
            # Log fold-specific metrics
            mlflow.log_metrics({
                f'fold_{fold_idx}_rmse': results['val_rmse'],
                f'fold_{fold_idx}_train_val_gap': results['train_val_gap']
            })
            
            # Prune trial if performance is poor
            trial.report(results['val_rmse'], fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Mean RMSE across folds
        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        
        mlflow.log_metrics({
            'mean_cv_rmse': mean_rmse,
            'std_cv_rmse': std_rmse
        })
        
        return mean_rmse


class CatBoostClassifierObjective:
    """
    Optuna objective for CatBoost classification.
    """
    
    def __init__(
        self,
        df,
        n_components_range: tuple = (50, 200),
        cv_start_cycle: int = 7
    ):
        self.df = df
        self.n_components_range = n_components_range
        self.cv_start_cycle = cv_start_cycle
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Returns:
            Mean F1-macro score across CV folds (maximize)
        """
        params = {
            'n_components': trial.suggest_int('n_components', *self.n_components_range),
            'iterations': trial.suggest_int('iterations', 500, 2000, step=100),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0)
        }
        
        mlflow.log_params(params)
        
        # Create CV splits
        cv_splitter = create_cv_splits_for_catboost(
            self.df,
            n_components=params['n_components'],
            start_cycle=self.cv_start_cycle,
            do_pca=True
        )
        
        # Train on each fold
        fold_f1_scores = []
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
            
            # Log fold metrics
            mlflow.log_metrics({
                f'fold_{fold_idx}_f1_macro': metrics['val_f1_macro'],
                f'fold_{fold_idx}_accuracy': metrics['val_accuracy']
            })
            
            # Prune if poor performance
            trial.report(metrics['val_f1_macro'], fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Mean F1 across folds
        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
        
        mlflow.log_metrics({
            'mean_cv_f1_macro': mean_f1,
            'std_cv_f1_macro': std_f1
        })
        
        return mean_f1


def run_lstm_optimization(
    df,
    n_trials: int = 100,
    timeout: int = 3600,
    study_name: str = 'lstm_optimization',
    mlflow_experiment_name: str = 'SWCNT_LSTM_Optimization'
) -> optuna.Study:
    """
    Run LSTM hyperparameter optimization with Optuna + MLflow.
    
    Args:
        df: Preprocessed dataframe
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        study_name: Optuna study name
        mlflow_experiment_name: MLflow experiment name
    
    Returns:
        Completed Optuna study
    """
    # Set MLflow experiment
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # Minimize RMSE
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=2,
            interval_steps=1
        )
    )
    
    # Create MLflow callback
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name='mean_cv_rmse'
    )
    
    # Create objective
    objective = LSTMObjective(df, cv_start_cycle=7)
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[mlflc],
        show_progress_bar=True,
        n_jobs=1  # Don't parallelize (MLflow logging issues)
    )
    
    # Log best parameters to MLflow
    with mlflow.start_run(run_name='best_lstm_params'):
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({
            'best_rmse': study.best_value,
            'n_trials': len(study.trials)
        })
    
    return study


def run_catboost_optimization(
    df,
    n_trials: int = 100,
    timeout: int = 3600,
    study_name: str = 'catboost_classification',
    mlflow_experiment_name: str = 'SWCNT_CatBoost_Classification'
) -> optuna.Study:
    """Run CatBoost classification optimization."""
    mlflow.set_experiment(mlflow_experiment_name)
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize F1
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=2
        )
    )
    
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name='mean_cv_f1_macro'
    )
    
    objective = CatBoostClassifierObjective(df, cv_start_cycle=7)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[mlflc],
        show_progress_bar=True,
        n_jobs=1
    )
    
    with mlflow.start_run(run_name='best_catboost_params'):
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({
            'best_f1_macro': study.best_value,
            'n_trials': len(study.trials)
        })
    
    return study