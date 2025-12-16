import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Iterator, Literal
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import joblib


# ============================================================================
# PREPROCESSING UTILITIES
# ============================================================================

def scale_n_PCA(
    train: np.ndarray,
    test: np.ndarray,
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale (MinMaxScaler) and apply PCA (optional) to train/test features.
    
    - Fits scaler and PCA on train only.
    - Applies the same transforms to test.
    
    Args:
        train: Training features (n_samples, n_features)
        test: Test features (n_samples, n_features)
        n_components: Number of PCA components (None = no PCA)
    
    Returns:
        Tuple of (transformed_train, transformed_test)
    """
    X_train = train.copy()
    X_test = test.copy()

    # 1) Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # 2) PCA (optional)
    if n_components is None or n_components < 1:
        return X_train, X_test
    
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


# ============================================================================
# SEQUENCE BUILDING FOR LSTM
# ============================================================================

def create_RNN_sequences(
    data: np.ndarray, 
    look_back: int = 50
) -> Tuple[np.ndarray, int]:
    """
    Build sliding-window sequences for RNN/LSTM.
    
    Args:
        data: (num_samples, num_features)
        look_back: window length
        
    Returns:
        sequences: (num_sequences, look_back, num_features)
        effective_look_back: actual window size used
    """
    num_samples, num_features = data.shape

    if num_samples < look_back + 10:
        raise ValueError(
            f"Not enough samples for look_back={look_back}: "
            f"num_samples={num_samples}")

    num_sequences = num_samples - look_back + 1
    sequences = np.zeros((num_sequences, look_back, num_features), dtype=float)

    for i in range(look_back):
        sequences[:, i] = data[i : i + num_sequences]

    return sequences, look_back


def build_sequences_for_df(
    df_sub: pd.DataFrame,
    feature_cols: int = 402,
    target_cols: List[str] = ['NO2', 'H2S', 'Acet'],
    look_back: int = 50,
    return_index: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Build RNN sequences for a dataframe subset grouped by 'gas_'.
    
    For each gas type, build sliding windows independently in time order.
    
    Args:
        df_sub: DataFrame with feature columns (0:feature_cols), 
                target columns, 'gas_' column
        feature_cols: number of feature columns (default 402)
        target_cols: list of target column names
        look_back: window length
        return_index: whether to return sequence indices
        
    Returns:
        X_seq: (N_seq_total, look_back, n_features)
        y_seq: (N_seq_total, n_targets)
        idx_seq: (N_seq_total,) or None
    """
    big_X, big_y, big_idx = [], [], []

    for gas in df_sub["gas_"].unique():
        df_gas = df_sub.loc[df_sub["gas_"] == gas].sort_index()  # Time-ordered

        X_gas = df_gas.iloc[:, :feature_cols].to_numpy()
        y_gas = df_gas.loc[:, target_cols].to_numpy()

        if len(X_gas) < look_back + 10:  # Safety margin
            raise ValueError(
                f"Gas '{gas}' has only {len(X_gas)} rows, "
                f"need >= {look_back + 10} to build sequences")

        seq_X, eff_lb = create_RNN_sequences(X_gas, look_back=look_back)
        # Align targets: one per sequence, at last time step of each window
        seq_y = y_gas[eff_lb - 1:, :]
        
        if return_index:
            idx = df_gas.index.to_numpy()
            seq_idx = idx[eff_lb - 1:]
            big_idx.append(seq_idx)

        if seq_X.shape[0] != seq_y.shape[0]:
            raise RuntimeError(
                f"Sequence/target size mismatch for gas={gas}: "
                f"{seq_X.shape[0]} vs {seq_y.shape[0]}")

        big_X.append(seq_X)
        big_y.append(seq_y)

    if not big_X:
        raise ValueError("No sequences could be built from the given subset.")

    X_seq = np.vstack(big_X)
    y_seq = np.vstack(big_y)
    
    if return_index:
        idx_seq = np.concatenate(big_idx) if big_idx else None
        return X_seq, y_seq, idx_seq

    return X_seq, y_seq, None


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CVFold:
    """Single cross-validation fold."""
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    train_index: Optional[np.ndarray] = None
    test_index: Optional[np.ndarray] = None
    train_sample_weights: Optional[np.ndarray] = None
    test_sample_weights: Optional[np.ndarray] = None
    fold_index: int = 0
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# TIME SERIES CV SPLITTER
# ============================================================================

class TimeSeriesCVSplitter:
    """
    Container for time-series CV folds.
    
    Supports:
    - Iteration, indexing
    - TensorFlow datasets
    - Save/load via joblib
    """

    def __init__(self, folds: List[CVFold]):
        self.folds = folds

    def __len__(self) -> int:
        return len(self.folds)

    def __getitem__(self, idx: int) -> CVFold:
        return self.folds[idx]

    def __iter__(self) -> Iterator[CVFold]:
        return iter(self.folds)

    def summary(self) -> str:
        """Print summary of all folds."""
        lines = [f"TimeSeriesCVSplitter: {len(self.folds)} folds\n"]
        for fold in self.folds:
            lines.append(
                f"  Fold {fold.fold_index}: "
                f"train={fold.train_X.shape[0]}, "
                f"test={fold.test_X.shape[0]}, "
                f"features={fold.train_X.shape[1:]}"
            )
        return "\n".join(lines)

    def to_tf_datasets(
        self, batch_size: int = 32
    ) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
        """
        Convert all folds to TensorFlow datasets.
        
        If sample weights are None, yields (X, y).
        If not None, yields (X, y, sample_weight).
        """
        datasets = []
        for fold in self.folds:
            if fold.train_sample_weights is None:
                train_ds = tf.data.Dataset.from_tensor_slices(
                    (fold.train_X, fold.train_y)
                )
            else:
                train_ds = tf.data.Dataset.from_tensor_slices(
                    (fold.train_X, fold.train_y, fold.train_sample_weights)
                )

            if fold.test_sample_weights is None:
                test_ds = tf.data.Dataset.from_tensor_slices(
                    (fold.test_X, fold.test_y)
                )
            else:
                test_ds = tf.data.Dataset.from_tensor_slices(
                    (fold.test_X, fold.test_y, fold.test_sample_weights)
                )

            train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            datasets.append((train_ds, test_ds))

        return datasets

    def save(self, path: str, compress: int = 3):
        """Save folds to disk using joblib."""
        joblib.dump(self.folds, path, compress=compress)

    @classmethod
    def load(cls, path: str) -> "TimeSeriesCVSplitter":
        """Load folds from disk."""
        folds = joblib.load(path)
        return cls(folds)


# ============================================================================
# FOLD GENERATORS
# ============================================================================


def create_catboost_folds(
    df: pd.DataFrame,
    feature_cols: int = 402,
    target_cols: List[str] = ['NO2', 'H2S', 'Acet'],
    n_components: Optional[int] = None,
    start_cycle: int = 7,
    test_size: int = 1,
    return_index: bool = False,
) -> TimeSeriesCVSplitter:
    """
    Create time-series CV folds for CatBoost (2D tabular data).
    
    Strategy:
    - Train on cycles <= start_cycle + i
    - Test on cycles [start_cycle + i + 1, start_cycle + i + test_size]
    - Each fold expands training set (no shuffling)
    
    Args:
        df: DataFrame with columns [0:feature_cols] as features,
            target_cols as targets, and 'meas_cycle' column
        feature_cols: number of feature columns
        target_cols: list of target column names
        n_components: PCA components (None = no PCA)
        start_cycle: minimum measurement cycle to start splits
        test_size: number of cycles to use for testing
        return_index: whether to return indices
        
    Returns:
        TimeSeriesCVSplitter with 2D folds
    """
    if 'meas_cycle' not in df.columns:
        raise ValueError("DataFrame must have 'meas_cycle' column")

    max_cycle = df['meas_cycle'].max()
    folds = []
    fold_idx = 0

    for split in range(start_cycle, max_cycle - test_size + 1):
        # Train: all cycles up to and including split
        train_mask = df['meas_cycle'] <= split
        # Test: next test_size cycles
        test_mask = (
            (df['meas_cycle'] > split) & 
            (df['meas_cycle'] <= split + test_size)
        )

        X_train_raw = df.loc[train_mask].iloc[:, :feature_cols].values
        X_test_raw = df.loc[test_mask].iloc[:, :feature_cols].values
        
        # Handle both single and multi-target cases
        if len(target_cols) == 1:
            y_train = df.loc[train_mask, target_cols[0]].values
            y_test = df.loc[test_mask, target_cols[0]].values
        else:
            y_train = df.loc[train_mask, target_cols].values
            y_test = df.loc[test_mask, target_cols].values

        train_index = df.index[train_mask].to_numpy() if return_index else None
        test_index = df.index[test_mask].to_numpy() if return_index else None

        # Apply scaling + PCA
        X_train, X_test = scale_n_PCA(
            X_train_raw, X_test_raw,
            n_components=n_components
        )

        fold = CVFold(
            train_X=X_train,
            train_y=y_train,
            test_X=X_test,
            test_y=y_test,
            train_index=train_index,
            test_index=test_index,
            fold_index=fold_idx,
            metadata={
                'train_cycles': f'<= {split}',
                'test_cycles': f'{split+1} to {split+test_size}',
                'n_train': X_train.shape[0],
                'n_test': X_test.shape[0]
            }
        )
        folds.append(fold)
        fold_idx += 1

    return TimeSeriesCVSplitter(folds)


def create_lstm_folds(
    df: pd.DataFrame,
    feature_cols: int = 402,
    target_cols: List[str] = ['NO2', 'H2S', 'Acet'],
    look_back: int = 50,
    n_components: Optional[int] = None,
    start_cycle: int = 7,
    test_size: int = 1,
    return_index: bool = False,
) -> TimeSeriesCVSplitter:
    """
    Create time-series CV folds for LSTM (3D sequential data).
    
    Strategy:
    - Same time-split logic as CatBoost
    - But builds 3D sequences per gas type
    - Applies scaling + PCA before sequence building
    
    Args:
        df: DataFrame with columns [0:feature_cols] as features,
            target_cols as targets, 'meas_cycle' and 'gas_' columns
        feature_cols: number of feature columns
        target_cols: list of target column names
        look_back: LSTM window length
        n_components: PCA components (None = no PCA)
        start_cycle: minimum measurement cycle to start splits
        test_size: number of cycles to use for testing
        return_index: whether to return indices
        
    Returns:
        TimeSeriesCVSplitter with 3D folds
    """
    if 'meas_cycle' not in df.columns:
        raise ValueError("DataFrame must have 'meas_cycle' column")
    if 'gas_' not in df.columns:
        raise ValueError("DataFrame must have 'gas_' column for LSTM sequences")

    max_cycle = df['meas_cycle'].max()
    folds = []
    fold_idx = 0

    for split in range(start_cycle, max_cycle - test_size + 1):
        # Train: all cycles up to and including split
        train_mask = df['meas_cycle'] <= split
        # Test: next test_size cycles
        test_mask = (
            (df['meas_cycle'] > split) & 
            (df['meas_cycle'] <= split + test_size)
        )

        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        # Apply scaling + PCA to raw features
        X_train_raw = df_train.iloc[:, :feature_cols].values
        X_test_raw = df_test.iloc[:, :feature_cols].values

        X_train_2d, X_test_2d = scale_n_PCA(
            X_train_raw, X_test_raw,
            n_components=n_components
        )
        
        # Build slim dfs: PCA features + required columns only
        keep_cols = ["meas_cycle", "gas_"] + target_cols
        df_train_small = pd.concat([
            pd.DataFrame(X_train_2d, index=df_train.index),
            df_train[keep_cols]
        ], axis=1)
        
        df_test_small = pd.concat([
            pd.DataFrame(X_test_2d, index=df_test.index),
            df_test[keep_cols]
        ], axis=1)

        # Build 3D sequences
        X_train_seq, y_train_seq, train_index = build_sequences_for_df(
            df_train_small,
            feature_cols=X_train_2d.shape[1],
            target_cols=target_cols,
            look_back=look_back,
            return_index=return_index
        )

        X_test_seq, y_test_seq, test_index = build_sequences_for_df(
            df_test_small,
            feature_cols=X_test_2d.shape[1],
            target_cols=target_cols,
            look_back=look_back,
            return_index=return_index
        )
        
        # If classifier mode: one-hot encode class_ (fit on TRAIN only)
        class_mapping = None
        if target_cols == ['class_']:
            enc = OneHotEncoder(sparse_output=False, handle_unknown='error')
            y_train_seq_2d = y_train_seq.reshape(-1, 1)
            y_test_seq_2d = y_test_seq.reshape(-1, 1)

            enc.fit(y_train_seq_2d)
            y_train_seq = enc.transform(y_train_seq_2d).astype(np.float32)
            y_test_seq = enc.transform(y_test_seq_2d).astype(np.float32)

            # Store mapping: class label -> onehot index
            cats = enc.categories_[0].tolist()
            class_mapping = {label: i for i, label in enumerate(cats)}

        fold = CVFold(
            train_X=X_train_seq,
            train_y=y_train_seq,
            test_X=X_test_seq,
            test_y=y_test_seq,
            train_index=train_index,
            test_index=test_index,
            fold_index=fold_idx,
            metadata={
                'train_cycles': f'<= {split}',
                'test_cycles': f'{split+1} to {split+test_size}',
                'n_train_sequences': X_train_seq.shape[0],
                'n_test_sequences': X_test_seq.shape[0],
                'look_back': look_back,
                'class_mapping': class_mapping,
            }
        )
        folds.append(fold)
        fold_idx += 1

    return TimeSeriesCVSplitter(folds)


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def create_time_series_folds(
    df: pd.DataFrame,
    model_type: Literal['catboost', 'lstm'],
    task_type: Literal['classifier', 'regressor'] = 'regressor',
    feature_cols: int = 402,
    target_cols: List[str] = ['NO2', 'H2S', 'Acet'],
    look_back: Optional[int] = 50,
    n_components: Optional[int] = None,
    start_cycle: int = 7,
    test_size: int = 1,
    return_index: bool = False,
) -> TimeSeriesCVSplitter:
    """
    Unified interface to create either CatBoost or LSTM folds.
    
    Args:
        df: DataFrame with features, targets, 'meas_cycle', and optionally 'gas_'
        model_type: 'catboost' for 2D or 'lstm' for 3D sequences
        task_type: 'classifier' or 'regressor'
        feature_cols: number of feature columns
        target_cols: list of target column names (for regressor)
        look_back: LSTM window length (ignored for CatBoost)
        n_components: PCA components (None = no PCA)
        start_cycle: minimum measurement cycle to start splits
        test_size: number of cycles to use for testing
        return_index: whether to return indices
        
    Returns:
        TimeSeriesCVSplitter with appropriate fold structure
        
    Example:
        >>> # For CatBoost regression
        >>> folds = create_time_series_folds(
        ...     df, model_type='catboost',
        ...     task_type='regressor',
        ...     target_cols=['NO2', 'H2S', 'Acet'],
        ...     n_components=50,
        ... )
        >>> 
        >>> # For LSTM classification
        >>> folds = create_time_series_folds(
        ...     df, model_type='lstm',
        ...     task_type='classifier',
        ...     look_back=30,
        ...     n_components=50,
        ... )
    """
    # Resolve target columns based on task type
    if task_type == 'classifier':
        resolved_target_cols = ['class_']
        if 'class_' not in df.columns:
            raise ValueError(
                "DataFrame must have 'class_' column for classifier mode")
    elif task_type == 'regressor':
        resolved_target_cols = target_cols
        for c in resolved_target_cols:
            if c not in df.columns:
                raise ValueError(f"Missing regression target column: {c}")
    else:
        raise ValueError("task_type must be 'classifier' or 'regressor'")

    # Create appropriate folds
    if model_type == 'catboost':
        return create_catboost_folds(
            df=df,
            feature_cols=feature_cols,
            target_cols=resolved_target_cols,
            n_components=n_components,
            start_cycle=start_cycle,
            test_size=test_size,
            return_index=return_index,
        )
    elif model_type == 'lstm':
        if look_back is None:
            raise ValueError("look_back must be specified for LSTM folds")
        return create_lstm_folds(
            df=df,
            feature_cols=feature_cols,
            target_cols=resolved_target_cols,
            look_back=look_back,
            n_components=n_components,
            start_cycle=start_cycle,
            test_size=test_size,
            return_index=return_index,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Use 'catboost' or 'lstm'.")