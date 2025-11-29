import numpy as np
import pandas as pd
from .loading import load_raw_gas
from .cleaning import apply_manual_trim

DP_PER_PULSE = 402  # from your experimental protocol and article

def cut_into_pulses(current: np.ndarray, dp_per_pulse: int = DP_PER_PULSE) -> np.ndarray:
    """
    Reshape a long I(t) vector into [n_cycles, dp_per_pulse].
    """
    if len(current) % dp_per_pulse != 0:
        raise ValueError(f"Length {len(current)} not divisible by {dp_per_pulse}")
    n_cycles = len(current) // dp_per_pulse
    return current.reshape(n_cycles, dp_per_pulse)

def build_basic_dataset(gas: str) -> pd.DataFrame:
    """
    Minimal placeholder: load one gas, trim, and reshape into cycles.
    Later we'll attach labels and merge gases.

    Returns
    -------
    pd.DataFrame
        Columns 0..(DP_PER_PULSE-1): current trace for each cycle.
    """
    df_raw = load_raw_gas(gas)
    df_clean = apply_manual_trim(df_raw, gas)

    if "I" not in df_clean.columns:
        raise KeyError("Expected column 'I' with current values in dataframe.")

    pulses = cut_into_pulses(df_clean["I"].values)
    return pd.DataFrame(pulses)
