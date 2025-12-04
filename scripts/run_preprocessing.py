"""
train_test_split.py

Time-series cross-validation splitting for SWCNT gas sensor data.

Key features:
- No shuffling (preserves temporal order)
- Expanding-window CV (train on cycles ≤n, test on cycle n+1)
- Separate sequence creation per gas (prevents cross-gas sequences in LSTM)
- Optional PCA dimensionality reduction
- Serialization support (pickle/joblib)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional
import pickle
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Optional: Only import TensorFlow if needed
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. `to_tf_datasets()` will not work.")


# ============================================================================
# Core Preprocessing Functions
# ============================================================================

def scale_and_pca(
    train: np.ndarray,
    test: np.ndarray,
    n_components: int,
    do_scaling: bool = True,
    do_pca: bool = True,
    scaler=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply scaling and/or PCA to train/test data.
    
    Args:
        train: Training data (n_samples, n_features)
        test: Test data (n_samples, n_features)
        n_components: Number of PCA components to keep
        do_scaling: Whether to apply MinMax scaling
        do_pca: Whether to apply PCA
        scaler: Sklearn scaler instance (default: MinMaxScaler)
    
    Returns:
        Transformed (train, test) arrays
    
    Note:
        Always fit on train, transform both train and test to prevent leakage.
    """
    # Scaling
    if do_scaling:
        if scaler is None:
            scaler = MinMaxScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    
    # PCA
    if do_pca:
        pca = PCA(n_components=n_components)
        train = pca.fit_transform(train)
        test = pca.transform(test)
    
    return train, test


def create_sequences(
    data: np.ndarray,
    look_back: int = 50
) -> tuple[np.ndarray, int]:
    """
    Create LSTM sequences from 2D array.
    
    Args:
        data: Input array (n_samples, n_features)
        look_back: Number of time steps per sequence
    
    Returns:
        sequences: Array of shape (n_sequences, look_back, n_features)
        actual_look_back: Adjusted look_back if input too small
    
    Example:
        data.shape = (100, 20)  # 100 samples, 20 features
        sequences.shape = (51, 50, 20)  # 51 sequences of 50 steps
    """
    num_samples, num_features = data.shape
    
    # Ensure look_back isn't too large
    max_look_back = num_samples * 3 // 4
    if look_back > max_look_back:
        warnings.warn(
            f"look_back={look_back} too large for {num_samples} samples. "
            f"Reducing to {max_look_back}."
        )
        look_back = max_look_back
    
    num_sequences = num_samples - look_back + 1
    sequences = np.zeros((num_sequences, look_back, num_features))
    
    # Vectorized sequence creation (more efficient than your original loop)
    for i in range(look_back):
        sequences[:, i, :] = data[i:i + num_sequences]
    
    return sequences, look_back


def create_sequences_by_gas(
    X: np.ndarray,
    y: np.ndarray,
    gas_labels: np.ndarray,
    look_back: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create LSTM sequences separately for each gas type.
    
    This prevents creating sequences that span across different gases,
    which would be invalid for LSTM training.
    
    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples, n_targets)
        gas_labels: Gas type for each sample (n_samples,)
        look_back: Sequence length
    
    Returns:
        X_sequences: (n_sequences, look_back, n_features)
        y_sequences: (n_sequences, n_targets)
    
    Example:
        If data has NO2, H2S, Acetone:
        - Create sequences within NO2 samples only
        - Create sequences within H2S samples only
        - Create sequences within Acetone samples only
        - Concatenate all sequences
    """
    all_sequences_X = []
    all_sequences_y = []
    
    # Process each gas separately
    for gas in np.unique(gas_labels):
        # Filter data for this gas
        gas_mask = gas_labels == gas
        X_gas = X[gas_mask]
        y_gas = y[gas_mask]
        
        # Create sequences for this gas
        sequences_X, actual_look_back = create_sequences(X_gas, look_back)
        
        # Target is the last value in each sequence
        # Shape: (n_sequences, n_targets)
        sequences_y = y_gas[actual_look_back - 1:]
        
        all_sequences_X.append(sequences_X)
        all_sequences_y.append(sequences_y)
    
    # Concatenate all gas sequences
    X_sequences = np.concatenate(all_sequences_X, axis=0)
    y_sequences = np.concatenate(all_sequences_y, axis=0)
    
    return X_sequences, y_sequences


# ============================================================================
# CV Fold Data Structures
# ============================================================================

@dataclass
class CVFold:
    """
    Single cross-validation fold for time-series data.
    
    Attributes:
        train_X: Training features (n_train, n_features) or (n_train, look_back, n_features)
        train_y: Training targets (n_train, n_targets)
        test_X: Test features (n_test, n_features) or (n_test, look_back, n_features)
        test_y: Test targets (n_test, n_targets)
        fold_index: Fold number (corresponds to measurement cycle)
        metadata: Additional info (n_components, look_back, train_cycles, etc.)
    """
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    fold_index: int
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate shapes after initialization."""
        assert self.train_X.shape[0] == self.train_y.shape[0], \
            f"Train X/y shape mismatch: {self.train_X.shape[0]} != {self.train_y.shape[0]}"
        assert self.test_X.shape[0] == self.test_y.shape[0], \
            f"Test X/y shape mismatch: {self.test_X.shape[0]} != {self.test_y.shape[0]}"
    
    @property
    def train_size(self) -> int:
        return self.train_X.shape[0]
    
    @property
    def test_size(self) -> int:
        return self.test_X.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features (last dimension)."""
        return self.train_X.shape[-1]


class TimeSeriesCVSplitter:
    """
    Container for time-series cross-validation folds.
    
    Features:
    - Iterate over folds: for fold in cv_splitter: ...
    - Index folds: cv_splitter[0]
    - Convert to TensorFlow datasets
    - Save/load to disk
    - Print summary
    """
    
    def __init__(self, folds: list[CVFold]):
        self.folds = folds
    
    def __len__(self) -> int:
        return len(self.folds)
    
    def __getitem__(self, idx: int) -> CVFold:
        return self.folds[idx]
    
    def __iter__(self) -> Iterator[CVFold]:
        return iter(self.folds)
    
    def summary(self) -> str:
        """Generate text summary of all folds."""
        lines = [f"TimeSeriesCVSplitter: {len(self)} folds"]
        lines.append("-" * 60)
        
        for fold in self.folds:
            lines.append(f"Fold {fold.fold_index}:")
            lines.append(f"  Train: {fold.train_X.shape}")
            lines.append(f"  Test:  {fold.test_X.shape}")
            lines.append(f"  Metadata: {fold.metadata}")
        
        return "\n".join(lines)
    
    def save(self, path: str, compress: bool = True):
        """
        Save folds to disk.
        
        Args:
            path: Output file path (.pkl or .joblib)
            compress: Use joblib compression (recommended for large arrays)
        """
        if compress and path.endswith('.joblib'):
            import joblib
            joblib.dump(self.folds, path, compress=3)
            print(f"✅ Saved {len(self)} folds to {path} (compressed)")
        else:
            with open(path, 'wb') as f:
                pickle.dump(self.folds, f)
            print(f"✅ Saved {len(self)} folds to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TimeSeriesCVSplitter':
        """Load folds from disk."""
        if path.endswith('.joblib'):
            import joblib
            folds = joblib.load(path)
        else:
            with open(path, 'rb') as f:
                folds = pickle.load(f)
        
        print(f"✅ Loaded {len(folds)} folds from {path}")
        return cls(folds)
    
    def to_tf_datasets(
        self,
        batch_size: int = 32,
        shuffle_buffer: Optional[int] = None
    ) -> list[tuple]:
        """
        Convert folds to TensorFlow datasets.
        
        Args:
            batch_size: Batch size for training
            shuffle_buffer: If provided, shuffle training data (NOT recommended for time-series!)
        
        Returns:
            List of (train_dataset, test_dataset) tuples
        
        Warning:
            Shuffling is disabled by default to preserve temporal order.
            Only enable if you've verified it's appropriate for your use case.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        datasets = []
        
        for fold in self.folds:
            # Create train dataset
            train_ds = tf.data.Dataset.from_tensor_slices((fold.train_X, fold.train_y))
            
            # ⚠️ IMPORTANT: No shuffling for time-series by default
            if shuffle_buffer is not None:
                warnings.warn(
                    "Shuffling time-series data may break temporal dependencies! "
                    "Only use if sequences are independent."
                )
                train_ds = train_ds.shuffle(buffer_size=shuffle_buffer)
            
            train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # Create test dataset (never shuffle)
            test_ds = tf.data.Dataset.from_tensor_slices((fold.test_X, fold.test_y))
            test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            datasets.append((train_ds, test_ds))
        
        return datasets


# ============================================================================
# Main Splitting Functions
# ============================================================================

def create_cv_splits_for_lstm(
    df: pd.DataFrame,
    look_back: int = 50,
    n_components: int = 100,
    start_cycle: int = 5,
    do_pca: bool = True,
    feature_cols: range = range(402),
    target_cols: range = range(402, 405),
    gas_col: str = 'gas',
    cycle_col: str = 'meas_cycle'
) -> TimeSeriesCVSplitter:
    """
    Create time-series CV splits for LSTM training.
    
    Strategy: Expanding window
    - Fold 0: Train on cycles ≤5, test on cycle 6
    - Fold 1: Train on cycles ≤6, test on cycle 7
    - ...
    
    Args:
        df: Preprocessed dataframe with features, targets, gas labels
        look_back: LSTM sequence length
        n_components: Number of PCA components
        start_cycle: First cycle to use for validation
        do_pca: Whether to apply PCA
        feature_cols: Columns for features (default: 0-401)
        target_cols: Columns for targets (default: 402-404)
        gas_col: Column name for gas labels
        cycle_col: Column name for measurement cycle
    
    Returns:
        TimeSeriesCVSplitter object
    """
    # Validate inputs
    required_cols = [gas_col, cycle_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    n_cycles = df[cycle_col].nunique()
    folds = []
    
    # Create one fold per cycle
    for split_idx in range(start_cycle, n_cycles - 1):
        # Split data by cycle
        train_mask = df[cycle_col] <= split_idx
        test_mask = df[cycle_col] == split_idx + 1
        
        # Extract features and targets
        X_train_raw = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_cols].values
        gas_train = df.loc[train_mask, gas_col].values
        
        X_test_raw = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_cols].values
        gas_test = df.loc[test_mask, gas_col].values
        
        # Apply scaling and PCA (fit on train only!)
        X_train, X_test = scale_and_pca(
            X_train_raw, X_test_raw,
            n_components=n_components,
            do_scaling=True,
            do_pca=do_pca
        )
        
        # Create LSTM sequences (grouped by gas to prevent cross-gas sequences)
        X_train_seq, y_train_seq = create_sequences_by_gas(
            X_train, y_train, gas_train, look_back
        )
        X_test_seq, y_test_seq = create_sequences_by_gas(
            X_test, y_test, gas_test, look_back
        )
        
        # Create fold
        fold = CVFold(
            train_X=X_train_seq,
            train_y=y_train_seq,
            test_X=X_test_seq,
            test_y=y_test_seq,
            fold_index=split_idx,
            metadata={
                'n_components': n_components,
                'look_back': look_back,
                'do_pca': do_pca,
                'train_cycles': list(range(start_cycle, split_idx + 1)),
                'test_cycle': split_idx + 1,
                'train_size_before_sequences': len(X_train_raw),
                'test_size_before_sequences': len(X_test_raw)
            }
        )
        folds.append(fold)
    
    return TimeSeriesCVSplitter(folds)


def create_cv_splits_for_catboost(
    df: pd.DataFrame,
    n_components: int = 100,
    start_cycle: int = 5,
    do_pca: bool = True,
    feature_cols: range = range(402),
    target_col: str = 'class_',
    cycle_col: str = 'meas_cycle'
) -> TimeSeriesCVSplitter:
    """
    Create time-series CV splits for CatBoost (no sequences needed).
    
    Args:
        df: Preprocessed dataframe
        n_components: Number of PCA components
        start_cycle: First cycle to use for validation
        do_pca: Whether to apply PCA
        feature_cols: Columns for features
        target_col: Column name for classification target
        cycle_col: Column name for measurement cycle
    
    Returns:
        TimeSeriesCVSplitter object
    """
    # Validate inputs
    required_cols = [target_col, cycle_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    n_cycles = df[cycle_col].nunique()
    folds = []
    
    for split_idx in range(start_cycle, n_cycles - 1):
        # Split data by cycle
        train_mask = df[cycle_col] <= split_idx
        test_mask = df[cycle_col] == split_idx + 1
        
        # Extract features and targets
        X_train_raw = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        
        X_test_raw = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values
        
        # Apply scaling and PCA
        X_train, X_test = scale_and_pca(
            X_train_raw, X_test_raw,
            n_components=n_components,
            do_scaling=True,
            do_pca=do_pca
        )
        
        # Create fold (no sequences for CatBoost)
        fold = CVFold(
            train_X=X_train,
            train_y=y_train,
            test_X=X_test,
            test_y=y_test,
            fold_index=split_idx,
            metadata={
                'n_components': n_components,
                'do_pca': do_pca,
                'train_cycles': list(range(start_cycle, split_idx + 1)),
                'test_cycle': split_idx + 1
            }
        )
        folds.append(fold)
    
    return TimeSeriesCVSplitter(folds)


# ============================================================================
# Usage Examples (for documentation)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage (run this file directly to test).
    """
    # Mock data for testing
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        **{i: np.random.randn(n_samples) for i in range(402)},
        402: np.random.rand(n_samples),  # NO2
        403: np.random.rand(n_samples),  # H2S
        404: np.random.rand(n_samples),  # Acetone
        'meas_cycle': np.repeat(range(10), n_samples // 10),
        'gas': np.tile(['NO2', 'H2S', 'Acetone'], n_samples // 3)[:n_samples],
        'class_': np.tile(['NO2', 'H2S', 'Acetone', 'air'], n_samples // 4)[:n_samples]
    })
    
    # Test LSTM splitting
    print("Creating LSTM CV splits...")
    cv_lstm = create_cv_splits_for_lstm(df, look_back=20, n_components=50, start_cycle=5)
    print(cv_lstm.summary())
    
    # Test CatBoost splitting
    print("\nCreating CatBoost CV splits...")
    cv_catboost = create_cv_splits_for_catboost(df, n_components=50, start_cycle=5)
    print(cv_catboost.summary())
    
    # Test serialization
    print("\nTesting save/load...")
    cv_lstm.save('test_cv.joblib', compress=True)
    cv_loaded = TimeSeriesCVSplitter.load('test_cv.joblib')
    print(f"Loaded {len(cv_loaded)} folds successfully!")