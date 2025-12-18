"""
lstm_model.py

LSTM model creation and training functions for time-series regression.
"""

import numpy as np
from optuna.integration import TFKerasPruningCallback
from optuna import Trial, TrialPruned
import tensorflow as tf
from tensorflow.keras import Input  # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense, Dropout  # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam  # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pyright: ignore[reportMissingImports]
from sklearn.metrics import root_mean_squared_error
from typing import Optional, Dict, Any

import logging
logger = logging.getLogger(__name__)

def build_lstm(
    input_shape: tuple,
    output_shape: int,
    n_layers: int = 1,
    n_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    loss: str = 'mse'
) -> tf.keras.Model:
    """
    Build LSTM model for multi-output regression.
    
    Args:
        input_shape: (look_back, n_features)
        output_shape: Number of output targets
        n_layers: Number of LSTM layers
        n_units: Units in first LSTM layer (halves each layer)
        dropout: Dropout rate for LSTM and recurrent dropout
        learning_rate: Optimizer learning rate
        loss: Loss function
    
    Returns:
        Compiled Keras model
    
    Example:
        >>> model = build_lstm(
        ...     input_shape=(50, 100),  # 50 timesteps, 100 features
        ...     output_shape=3,          # 3 gas concentrations
        ...     n_layers=2,
        ...     n_units=128
        ... )
    """
    model = Sequential(name='LSTM_MultiOutput')
    model.add(Input(shape = input_shape))
    
    # Add LSTM layers
    for layer_idx in range(n_layers, 0, -1):
        return_sequences = layer_idx > 1  # Only return sequences for non-final layers
        
        model.add(LSTM(
            units=n_units,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=dropout,
            name=f'lstm_{n_layers - layer_idx + 1}',
            unroll=False  # Prevents excessive retracing
        ))
        
        # Halve units for next layer (minimum 8)
        n_units = max(n_units // 2, 8)
    
    # Output layer
    model.add(Dense(units=output_shape, name='output'))
    
    # Compile model
    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['mae', 'mse']
    )
    
    return model


def train_lstm(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 30,
    min_delta: float = 0.001,
    verbose: int = 0,
    reduce_lr: bool = True,
    trial: Optional[Trial] = None, 
    fold_idx: Optional[int] = None 
) -> Dict[str, Any]:
    """
    Train LSTM model with optional Optuna pruning callback.
    
    Args:
        ... (existing args)
        trial: Optuna trial for pruning (optional)
        fold_idx: Current fold index for reporting (optional)
    """
    # Existing callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1 if verbose > 0 else 0
        )
    ]
    
    if reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                verbose=1 if verbose > 0 else 0
            )
        )
    
    # ✅ ADD OPTUNA PRUNING CALLBACK
    if trial is not None:
        callbacks.append(
            TFKerasPruningCallback(
                trial,
                monitor='val_loss'
            )
        )
    
    # ✅ WRAP TRAINING IN TRY-EXCEPT
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False
        )
        
        # Evaluate on validation set
        y_pred = model.predict(X_val, batch_size=batch_size, verbose=0)
        val_rmse = root_mean_squared_error(
            y_val, y_pred, 
            multioutput='uniform_average'
        )
        
        # Calculate train/val gap
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        stable_slice = slice(-patience, None) if len(train_loss) > patience else slice(None)
        train_val_gap = np.mean(np.abs(
            np.array(train_loss[stable_slice]) - np.array(val_loss[stable_slice])
        ))
        
        return {
            'history': history.history,
            'best_epoch': len(train_loss),
            'val_rmse': float(val_rmse),
            'train_val_gap': float(train_val_gap),
            'final_train_loss': float(train_loss[-1]),
            'final_val_loss': float(val_loss[-1]),
            'pruned': False 
        }
    
    # ✅ CATCH PRUNING EXCEPTION
    except TrialPruned:
        logger.info(f"Trial pruned at fold {fold_idx}")
        return {
            'pruned': True, 
            'val_rmse': float('inf')
        }


def evaluate_lstm(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate LSTM model on test set.
    
    Returns:
        Dictionary with RMSE metrics per output and overall
    """
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
    
    # Overall RMSE
    rmse_overall = root_mean_squared_error(
        y_test, y_pred, 
        multioutput='uniform_average'
    )
    
    # Per-output RMSE
    rmse_per_output = root_mean_squared_error(
        y_test, y_pred, 
        multioutput='raw_values'
    )
    
    metrics = {
        'test_rmse': float(rmse_overall),
        'test_rmse_no2': float(rmse_per_output[0]),
        'test_rmse_h2s': float(rmse_per_output[1]),
        'test_rmse_acet': float(rmse_per_output[2])
    }
    
    return metrics