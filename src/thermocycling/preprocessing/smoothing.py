import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

#from data.assemble import DP_PER_PULSE

from typing import Callable, Any, List
# from statsmodels.nonparametric.smoothers_lowess import lowess

def Exp(data: pd.Series, alpha: float) -> np.ndarray:
    """
        Apply forward-backward exponential smoothing to 1D data.
    """
    data = data.to_numpy()
    if alpha >= 1 or alpha <= 0:
        raise ValueError("Enter alpha such that 0 < alpha < 1")
    n = len(data)
    smoothed_data = np.zeros_like(data, dtype=float)

    # Forward pass
    smoothed_data[0] = data[0]
    for i in range(1, n):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]

    # Backward pass
    for i in range(n - 2, -1, -1):
        smoothed_data[i] = alpha * smoothed_data[i] + (1 - alpha) * smoothed_data[i + 1]

    return smoothed_data

def Exp_pd(data: pd.Series, alpha: float, adjust: bool = False) -> np.ndarray:
    """
    Args:
        alpha: float between 0 and 1
        adjust: whether to apply a built-in adjust 
    """
    forward = data.ewm(alpha=alpha, adjust=adjust).mean()
    backward = forward[::-1].ewm(alpha=alpha, adjust=adjust).mean()
    
    return backward.to_numpy()[::-1]

def Savitzky_Golay(data: pd.Series, **kwargs) -> np.ndarray:
    kwargs.setdefault('window_length', 300)
    kwargs.setdefault('polyorder', 3)
    kwargs.setdefault('mode', 'mirror')
    return savgol_filter(x = data.to_numpy(), **kwargs)

def dedrift(df: pd.DataFrame, envelope_ind: int | List[int], dedrift_func: Callable[..., Any], **kwargs) -> pd.DataFrame:
    
    data = df.iloc[:, :402]
    
    if type(envelope_ind) == int: envelope_ind = [envelope_ind]
    envelopes = {}
    for ind in envelope_ind:
        if ind < 0 or ind >= 402:
            raise IndexError(f"Envelope index {ind} out of range for 402={402}")
        series = data.iloc[:, ind]
        envelopes[ind] = dedrift_func(series, **kwargs)
    envelope = pd.DataFrame(envelopes).values.mean(axis=1)  # calculating final envelope as the mean of all envelopes
    dedrifted_data = data - envelope.reshape(-1,1)
    if df.shape[1] > 402:
        dedrifted_df = pd.concat([dedrifted_data, df.iloc[:,402:]], axis=1)
    else: raise ValueError(f"Wrong dataframe passed! No target data included, df.shape[1] = {df.shape[1]}")
    
    return dedrifted_df
