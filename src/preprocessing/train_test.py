import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Iterator

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib


"""TODO: 
    1. disable sample weighting (this is excessive!)
    2. clean up the code logic. I think the current pipeline might be optimized
    3. REMAKE create_RNN_sequences_for_multiple_gases
    4. REMAKE train_test_TS_class in a OOP way
    """
def scale_n_PCA(
    train: np.ndarray,
    test: np.ndarray,
    n_components: Optional[int] = None,
    scale: bool = True,
    scaler: Optional[MinMaxScaler] = None,
    do_PCA: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale (optional) and apply PCA (optional) to train/test features.

    - Fits scaler and PCA on train only.
    - Applies the same transforms to test.
    - Always returns transformed train, test.
    """
    X_train = train.copy()
    X_test = test.copy()

    # 1) Scaling
    if scale:
        if scaler is None:
            scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # 2) PCA
    if do_PCA:
        if n_components is None:
            raise ValueError("n_components must be specified when do_PCA=True")

        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test

    
def create_RNN_sequences(data: np.ndarray, look_back: int = 50) -> Tuple[np.ndarray, int]:
    """
    Build sliding-window sequences for RNN/LSTM.

    Args:
        data: (num_samples, num_features)
        look_back: window length.

    Returns:
        sequences: (num_sequences, look_back, num_features)
        effective_look_back: possibly reduced look_back if data is too short.
    """
    num_samples, num_features = data.shape

    # Adjust look_back if it's too large
    if num_samples // 4 * 3 < look_back:
        look_back = int(num_samples // 4 * 3)
        print(
            f"Look back too large, num_samples={num_samples}. "
            f"Lookback set to {look_back}"
        )

    num_sequences = num_samples - look_back + 1
    sequences = np.zeros((num_sequences, look_back, num_features))

    for i in range(look_back):
        sequences[:, i] = data[i : num_samples - look_back + 1 + i]

    return sequences, look_back

def build_sequences_for_df(
    df_sub: pd.DataFrame,
    feature_cols: int = 402,
    target_cols: List[str] = ['NO2', 'H2S', 'Acet'],
    look_back: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build RNN sequences for a dataframe subset that already has 'gas_'.

    For each gas_, build sliding windows independently in time order.

    Args:
        df_sub: subset with columns feature_cols + target_cols + 'gas_' + 'class_'.
        target_cols: target column names (e.g. ['NO2', 'H2S', 'Acet'] for regression or ['class_'] for classification).
        look_back: window length.

    Returns:
        X_seq: (N_seq_total, look_back, n_features)
        y_seq: (N_seq_total, n_targets)
    """
    big_X, big_y = [], []

    for gas in df_sub["gas_"].unique():
        df_gas = df_sub.loc[df_sub["gas_"] == gas].sort_values(["meas_cycle", "Time"])

        X_gas = df_gas.iloc[:,:feature_cols].to_numpy()
        y_gas = df_gas.loc[target_cols].to_numpy()

        if len(X_gas) <= 1:
            continue  # not enough data

        seq_X, eff_lb = create_RNN_sequences(X_gas, look_back=look_back)
        # align targets: one per sequence, at last time step of each window
        seq_y = y_gas[eff_lb - 1 :, :]

        if seq_X.shape[0] != seq_y.shape[0]:
            raise RuntimeError(
                f"Sequence/target size mismatch for gas={gas}: "
                f"{seq_X.shape[0]} vs {seq_y.shape[0]}"
            )

        big_X.append(seq_X)
        big_y.append(seq_y)

    if not big_X:
        raise ValueError("No sequences could be built from the given subset.")

    X_seq = np.vstack(big_X)
    y_seq = np.vstack(big_y)

    return X_seq, y_seq

@dataclass
class CVFold:
    """Single cross-validation fold."""
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    train_sample_weights: Optional[np.ndarray] = None
    test_sample_weights: Optional[np.ndarray] = None
    fold_index: int = 0
    metadata: Optional[Dict] = None


class TimeSeriesCVSplitter:
    """
    Container for time-series CV folds.

    Supports iteration, indexing, TensorFlow datasets, and save/load.
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
        joblib.dump(self.folds, path, compress=compress)

    @classmethod
    def load(cls, path: str) -> "TimeSeriesCVSplitter":
        folds = joblib.load(path)
        return cls(folds)


# think about the ways to optimize this process
### IMPORTANT ###
# the data already has a gas_ column!
def create_RNN_sequences_for_multiple_gases(X: np.ndarray, y: np.ndarray, gas_type: np.ndarray, look_back: int = 50, to_weight: bool = True) -> (np.ndarray, np.ndarray):
    df_ = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis = 1, ignore_index=True).assign(gas_ = gas_type)
    big_sequence_X, big_sequence_y = [], []
    sample_weights = []
    for gas in df_.gas_.unique():
        sequence_X, new_look_back = create_RNN_sequences(data = df_.loc[df_['gas_'] == gas].iloc[:,:X.shape[1]].values, look_back=look_back)
        sequence_y = df_.loc[df_['gas_']==gas].iloc[new_look_back - 1:, X.shape[1]:X.shape[1] + y.shape[1]].values
        big_sequence_X.extend(list(sequence_X))
        big_sequence_y.extend(list(sequence_y))
        #sample_weights.extend([sample_weight(gas, look_back = new_look_back, to_weight=to_weight)]*len(sequence_X))  # disable sample weiting
    return np.array(big_sequence_X), np.array(big_sequence_y) #, np.array(sample_weights)

# remake this similar to RNN train_test rework
def train_test_TS_class(df_: pd.DataFrame, n_components: int = 5, start: int = 1) -> (list, list, list, list):
    '''start: int, meas_cycle to start with
    returns 4 lists for train/test x/y. Each element = split'''
    if not ('meas_cycle' or 'class_') in df_.columns:
        raise ValueError("wrong dataframe: no meas_cycle or class_ column")
    else:
        n_splits = len(df_.meas_cycle.unique())
        train_X, test_X, train_y, test_y = [], [], [], []
        for split in range(start,n_splits-1):  # starting from 1 cause the first meas cycle was cut
            # PCA
            train = df_.loc[df_['meas_cycle']<=split].iloc[:,:402].values
            test = df_.loc[df_['meas_cycle']==split+1].iloc[:,:402].values
            train_pca, test_pca = scale_n_PCA(train, test, n_components=n_components, scale=True, scaler=MinMaxScaler(), do_PCA=True)
            train_X.append(train_pca)
            train_y.append(df_.loc[df_['meas_cycle']<=split].loc[:,"class_"].values)
            test_X.append(test_pca)
            test_y.append(df_.loc[df_['meas_cycle']==split+1].loc[:,"class_"].values)
        return train_X, test_X, train_y, test_y

# Claude draft

@dataclass
class CVFold:
    """Single cross-validation fold."""
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    train_sample_weights: np.ndarray
    test_sample_weights: np.ndarray
    fold_index: int
    metadata: dict

class TimeSeriesCVSplitter:
    """
    Manages time-series cross-validation splits.
    
    Supports iteration, indexing, and conversion to different formats.
    """
    
    def __init__(self, folds: list[CVFold]):
        self.folds = folds
    
    def __len__(self) -> int:
        return len(self.folds)
    
    def __getitem__(self, idx: int) -> CVFold:
        return self.folds[idx]
    
    def __iter__(self) -> Iterator[CVFold]:
        return iter(self.folds)
    
    def get_fold(self, idx: int) -> CVFold:
        """Get specific fold."""
        return self.folds[idx]
    
    def to_tf_datasets(self, batch_size: int = 32) -> list[tuple[tf.data.Dataset, tf.data.Dataset]]:
        """Convert all folds to TensorFlow datasets."""
        datasets = []
        for fold in self.folds:
            train_ds = tf.data.Dataset.from_tensor_slices((
                fold.train_X, fold.train_y, fold.train_sample_weights
            )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            test_ds = tf.data.Dataset.from_tensor_slices((
                fold.test_X, fold.test_y, fold.test_sample_weights
            )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            datasets.append((train_ds, test_ds))
        
        return datasets
    
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
    
    def save(self, path: str):
        """Save all folds to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.folds, f)
    
    @classmethod
    def load(cls, path: str) -> 'TimeSeriesCVSplitter':
        """Load folds from disk."""
        import pickle
        with open(path, 'rb') as f:
            folds = pickle.load(f)
        return cls(folds)

def train_test_RNN_cv(df_: pd.DataFrame, look_back: int = 50, n_components: int = 5, start: int = 5, do_PCA: bool = True, to_weight: bool = False) -> TimeSeriesCVSplitter:
    folds = []
    for split_idx in range(start, n_splits - 1):
        n_splits = len(df_.meas_cycle.unique())
        train_X, test_X, train_y, test_y = [], [], [], []
        for split in range(start,n_splits-1):  # starting from 1 cause the first meas cycle was cut
            train_y_ = df_.loc[df_['meas_cycle']<=split].iloc[:,402:405].values
            test_y_ = df_.loc[df_['meas_cycle']==split+1].iloc[:,402:405].values  # df_['meas_cycle']==split+1  df_['meas_cycle']>split
            train_gas = df_.loc[df_['meas_cycle']<=split].loc[:, 'gas'].values
            test_gas = df_.loc[df_['meas_cycle']==split+1].loc[:, 'gas'].values  # df_['meas_cycle']==split+1

            # PCA
            train = df_.loc[df_['meas_cycle']<=split].iloc[:,:402].values
            test = df_.loc[df_['meas_cycle']>split].iloc[:,:402].values  # df_['meas_cycle']==split+1
            train_pca, test_pca = scale_n_PCA(train, test, n_components=n_components, scale=True, scaler=MinMaxScaler(), do_PCA=do_PCA)

            # careful splitting
            train_X_, train_y__ = create_RNN_sequences_for_multiple_gases(X = train_pca, y = train_y_, gas_type = train_gas, look_back = look_back, to_weight=to_weight)
            train_X.append(train_X_)
            train_y.append(train_y__)
            
            test_X_, test_y__ = create_RNN_sequences_for_multiple_gases(X = test_pca, y = test_y_, gas_type = test_gas, look_back = look_back, to_weight=to_weight)
            test_X.append(test_X_)
            test_y.append(test_y__)
        
        fold = CVFold(
            train_X=train_X_,
            train_y=train_y_,
            test_X=test_X_,
            test_y=test_y_,
            #train_sample_weights=train_sw,
            #test_sample_weights=test_sw,
            fold_index=split_idx,
            metadata={
                'n_components': n_components,
                'look_back': look_back,
                'train_cycles': list(range(0, split_idx + 1)),
                'test_cycles': [split_idx + 1]
            }
        )
        folds.append(fold)
    
    return TimeSeriesCVSplitter(folds)

# Clean usage:
# cv_splitter = train_test_RNN_cv(df, look_back=50, n_components=100)
# print(cv_splitter.summary())

# # For LSTM (TensorFlow):
# tf_datasets = cv_splitter.to_tf_datasets(batch_size=128)
# for fold, (train_ds, test_ds) in enumerate(tf_datasets):
#     model.fit(train_ds, epochs=100, validation_data=test_ds)

# # For CatBoost (numpy arrays):
# for fold in cv_splitter:
#     model.fit(fold.train_X, fold.train_y, sample_weight=fold.train_sample_weights)
#     score = model.score(fold.test_X, fold.test_y)