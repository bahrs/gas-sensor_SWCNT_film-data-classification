import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    mean_squared_error, mean_absolute_error
)
from typing import Dict, Any, Literal, Optional

"""
catboost_model.py

CatBoost model wrapper for classification and regression.
"""


def build_catboost_classifier(
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 3.0,
    loss_function: str = 'MultiClass',
    eval_metric: str = 'TotalF1',
    verbose: bool = False,
    **kwargs
) -> CatBoostClassifier:
    """
    Build CatBoost classifier.
    
    Args:
        iterations: Number of boosting iterations
        depth: Tree depth
        learning_rate: Learning rate
        l2_leaf_reg: L2 regularization
        loss_function: Loss function
        eval_metric: Evaluation metric
        verbose: Print training progress
        **kwargs: Additional CatBoost parameters
    
    Returns:
        CatBoost classifier instance
    """
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function=loss_function,
        eval_metric=eval_metric,
        verbose=verbose,
        random_seed=42,
        **kwargs
    )
    
    return model


def train_catboost_classifier(
    model: CatBoostClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train CatBoost classifier with early stopping.
    
    Returns:
        Dictionary with training metrics
    """
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        plot=False
    )
    
    # Get best iteration
    best_iteration = model.get_best_iteration()
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    
    metrics = {
        'best_iteration': best_iteration,
        'val_f1_macro': f1_score(y_val, y_pred, average='macro'),
        'val_accuracy': accuracy_score(y_val, y_pred)
    }
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_val, y_pred, average=None)
    class_names = model.classes_
    for class_name, f1 in zip(class_names, f1_per_class):
        metrics[f'val_f1_{class_name}'] = float(f1)
    
    return metrics


def build_catboost_regressor(
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 3.0,
    loss_function: str = 'MultiRMSE',
    verbose: bool = False,
    **kwargs
) -> CatBoostRegressor:
    """Build CatBoost regressor."""
    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function=loss_function,
        verbose=verbose,
        random_seed=42,
        **kwargs
    )
    
    return model


def train_catboost_regressor(
    model: CatBoostRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """Train CatBoost regressor."""
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        plot=False
    )
    
    best_iteration = model.get_best_iteration()
    y_pred = model.predict(X_val)
    
    # Overall metrics
    rmse_overall = mean_squared_error(y_val, y_pred, squared=False, multioutput='uniform_average')
    mae_overall = mean_absolute_error(y_val, y_pred, multioutput='uniform_average')
    
    # Per-output metrics
    rmse_per_output = mean_squared_error(y_val, y_pred, squared=False, multioutput='raw_values')
    
    metrics = {
        'best_iteration': best_iteration,
        'val_rmse': float(rmse_overall),
        'val_mae': float(mae_overall),
        'val_rmse_no2': float(rmse_per_output[0]),
        'val_rmse_h2s': float(rmse_per_output[1]),
        'val_rmse_acet': float(rmse_per_output[2])
    }
    
    return metrics