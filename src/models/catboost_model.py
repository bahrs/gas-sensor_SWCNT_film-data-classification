"""
catboost_model.py

CatBoost model wrapper for classification and regression tasks.
"""

import numpy as np
from optuna import Trial, TrialPruned
from optuna.integration import CatBoostPruningCallback
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    root_mean_squared_error, mean_absolute_error
)
from typing import Dict, Any, Optional

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# CLASSIFICATION
# ============================================================================

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
    verbose: bool = False,
    trial: Optional[Trial] = None,  # ← ADD THIS
    fold_idx: Optional[int] = None  # ← ADD THIS
) -> Dict[str, Any]:
    """
    Train CatBoost classifier with optional Optuna pruning.
    
    Args:
        ... (existing args)
        trial: Optuna trial for pruning (optional)
        fold_idx: Current fold index (optional)
    """
    
    # ✅ CREATE PRUNING CALLBACK IF TRIAL PROVIDED
    callbacks = []
    if trial is not None:
        # CatBoost pruning callback
        pruning_callback = CatBoostPruningCallback(
            trial,
            metric='TotalF1:use_weights=false'  # Match your eval_metric
        )
        callbacks.append(pruning_callback)
    
    # ✅ WRAP IN TRY-EXCEPT
    try:
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            plot=False,
            callbacks=callbacks  # ← ADD THIS
        )
        
        # Get best iteration
        best_iteration = model.get_best_iteration()
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        
        metrics = {
            'best_iteration': best_iteration,
            'val_f1_macro': f1_score(y_val, y_pred, average='macro'),
            'val_accuracy': accuracy_score(y_val, y_pred),
            'pruned': False  # ← ADD THIS
        }
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_val, y_pred, average=None)
        class_names = model.classes_
        for class_name, f1 in zip(class_names, f1_per_class):
            metrics[f'val_f1_{class_name}'] = float(f1)
        
        return metrics
    
    # ✅ CATCH PRUNING EXCEPTION
    except TrialPruned:
        logger.info(f"Trial pruned at fold {fold_idx}")
        return {
            'pruned': True,
            'val_f1_macro': 0.0,
            'val_accuracy': 0.0
        }


# ============================================================================
# REGRESSION
# ============================================================================

def build_catboost_regressor(
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 3.0,
    loss_function: str = 'MultiRMSE',
    verbose: bool = False,
    **kwargs
) -> CatBoostRegressor:
    """
    Build CatBoost regressor.
    
    Args:
        iterations: Number of boosting iterations
        depth: Tree depth
        learning_rate: Learning rate
        l2_leaf_reg: L2 regularization
        loss_function: Loss function (use 'MultiRMSE' for multi-target)
        verbose: Print training progress
        **kwargs: Additional CatBoost parameters
    
    Returns:
        CatBoost regressor instance
    """
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
    verbose: bool = False,
    trial: Optional[Trial] = None,  # ← ADD THIS
    fold_idx: Optional[int] = None  # ← ADD THIS
) -> Dict[str, Any]:
    """
    Train CatBoost regressor with optional Optuna pruning.
    """
    
    # ✅ CREATE PRUNING CALLBACK
    callbacks = []
    if trial is not None:
        pruning_callback = CatBoostPruningCallback(
            trial,
            metric='MultiRMSE'  # Match your loss_function
        )
        callbacks.append(pruning_callback)
    
    # ✅ WRAP IN TRY-EXCEPT
    try:
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            plot=False,
            callbacks=callbacks  # ← ADD THIS
        )
        
        best_iteration = model.get_best_iteration()
        y_pred = model.predict(X_val)
        
        # Overall metrics
        rmse_overall = root_mean_squared_error(
            y_val, y_pred,
            multioutput='uniform_average'
        )
        mae_overall = mean_absolute_error(
            y_val, y_pred, 
            multioutput='uniform_average'
        )
        
        # Per-output metrics
        rmse_per_output = root_mean_squared_error(
            y_val, y_pred, 
            multioutput='raw_values'
        )
        
        metrics = {
            'best_iteration': best_iteration,
            'val_rmse': float(rmse_overall),
            'val_mae': float(mae_overall),
            'val_rmse_no2': float(rmse_per_output[0]),
            'val_rmse_h2s': float(rmse_per_output[1]),
            'val_rmse_acet': float(rmse_per_output[2]),
            'pruned': False  # ← ADD THIS
        }
        
        return metrics
    
    # ✅ CATCH PRUNING EXCEPTION
    except TrialPruned:
        logger.info(f"Trial pruned at fold {fold_idx}")
        return {
            'pruned': True,
            'val_rmse': float('inf'),
            'val_mae': float('inf')
        }