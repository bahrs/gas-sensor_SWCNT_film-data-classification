"""
lstm_model.py

LSTM model creation and training functions for time-series regression.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense, Dropout  # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam  # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pyright: ignore[reportMissingImports]
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Any


def build_lstm(
    input_shape: tuple,
    output_shape: int,
    n_layers: int = 1,
    n_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    loss: str = 'mean_squared_error'
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
    
    # Add LSTM layers
    for layer_idx in range(n_layers, 0, -1):
        return_sequences = layer_idx > 1  # Only return sequences for non-final layers
        
        model.add(LSTM(
            units=n_units,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=dropout,
            name=f'lstm_{n_layers - layer_idx + 1}'
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
    reduce_lr: bool = True
) -> Dict[str, Any]:
    """
    Train LSTM model with early stopping.
    
    Args:
        model: Compiled Keras model
        X_train: Training sequences (n_samples, look_back, n_features)
        y_train: Training targets (n_samples, n_outputs)
        X_val: Validation sequences
        y_val: Validation targets
        epochs: Maximum epochs
        batch_size: Batch size
        patience: Early stopping patience
        min_delta: Minimum change to qualify as improvement
        verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        reduce_lr: Whether to reduce LR on plateau
    
    Returns:
        Dictionary with:
        - history: Training history
        - best_epoch: Epoch with best validation loss
        - val_rmse: Validation RMSE
        - train_val_gap: Gap between train and val loss
    """
    # Callbacks
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
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False  # ⚠️ CRITICAL: Don't shuffle time-series!
    )
    
    # Evaluate on validation set
    y_pred = model.predict(X_val, batch_size=batch_size, verbose=0)
    val_rmse = mean_squared_error(
        y_val, y_pred, 
        squared=False, 
        multioutput='uniform_average'
    )
    
    # Calculate train/val gap (stability metric)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Focus on last `patience` epochs (most stable part)
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
        'final_val_loss': float(val_loss[-1])
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
    rmse_overall = mean_squared_error(
        y_test, y_pred, 
        squared=False, 
        multioutput='uniform_average'
    )
    
    # Per-output RMSE
    rmse_per_output = mean_squared_error(
        y_test, y_pred, 
        squared=False, 
        multioutput='raw_values'
    )
    
    metrics = {
        'test_rmse': float(rmse_overall),
        'test_rmse_no2': float(rmse_per_output[0]),
        'test_rmse_h2s': float(rmse_per_output[1]),
        'test_rmse_acet': float(rmse_per_output[2])
    }
    
    return metrics