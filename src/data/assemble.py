import numpy as np
import pandas as pd
from typing import Callable, Any, List

from .loading import load_gas_data
from .cleaning import apply_manual_trim
from .paths import PROCESSED_FILE_MAP
from preprocessing.smoothing import dedrift

DP_PER_PULSE = 402  # from experimental protocol (see the article)
MAX_FLOW = 50  # max flow through MFC, sccm (see the article)
VESSEL_CON = 100  # ppm, see article for details

# TODO: finalize the calculation for the actual raw gas


def cut_into_pulses(current: np.ndarray, dp_per_pulse: int = DP_PER_PULSE) -> np.ndarray:
    """
    Reshape a long I(t) vector into [n_cycles, dp_per_pulse].
    """
    if len(current) % dp_per_pulse != 0:
        raise ValueError(f"Length {len(current)} not divisible by {dp_per_pulse}")
    n_cycles = len(current) // dp_per_pulse
    return current.reshape(n_cycles, dp_per_pulse)

def build_basic_dataset(gas: str, gas_raw: bool = False) -> pd.DataFrame:
    """
    Args:
        gas: str; one of NO2, H2S, Acet, NO2_2
        gas_raw: whether to include true gas concentration values for each pulse
    Returns
    -------
    pd.DataFrame
        Columns 0..(DP_PER_PULSE-1): current trace for each cycle.
    """
    #df_raw = load_gas_data(gas, raw=True)
    #df_clean = apply_manual_trim(df_raw, gas)
    df_clean = load_gas_data(gas, raw=False)    
          
    if "I" not in df_clean.columns:
        raise KeyError("Expected column 'I' with current values in dataframe.")
    if gas == 'NO2_2': gas = 'NO2'

    pulses = pd.DataFrame(data = cut_into_pulses(df_clean["I"].values), 
                          index = df_clean.index[::DP_PER_PULSE])
    meas_cycle_col = df_clean["meas_cycle"].values[::DP_PER_PULSE]
    gas_col = df_clean["MFC_target"].values[::DP_PER_PULSE]
    if gas_raw:
        # expected = df_clean["MFC_target"].values
        # actual = [(MAX_FLOW*(100 - target_err)*VESSEL_CONC/(MAX_FLOW*(100 - target_err) + MAX_FLOW*(100 - carrier_err) + 50) 
        #     for target_err, carrier_err in zip(df_clean["flow_target_error"].values, df_clean["flow_carrier_error"].values)
        # ]
        # gas_col = np.array(actual).reshape(len(actual) // DP_PER_PULSE, DP_PER_PULSE).mean(axis = 1)
        pass
    DF = pulses.assign(
        NO2 = [0] * len(meas_cycle_col),
        H2S = [0] * len(meas_cycle_col),
        Acet = [0] * len(meas_cycle_col),
        meas_cycle = meas_cycle_col,
        gas_ = gas,
        class_ = ['air' if conc == 0 else gas for conc in gas_col]             
    )
    if gas == 'NO2_2': gas = "NO2"  # a workaround to include NO2 recorded later
    DF[gas] = gas_col
    return DF
    
def full_dataset(dedrifting_func: Callable[..., Any], envelope_ind: int | List[int] , gas_raw: bool = False, **kwargs) -> pd.DataFrame:
    """
    Args:
        gas_raw: bool
    Returns:
    --------
    pd.DataFrame
    """
    DF = pd.DataFrame()
    for gas in ["NO2", "H2S", "Acet"]:
        gas_df = build_basic_dataset(gas, gas_raw)
        dedrifted = dedrift(df = gas_df, envelope_ind=envelope_ind, dedrift_func= dedrifting_func, **kwargs)
        DF = pd.concat([DF, dedrifted])
    return DF
