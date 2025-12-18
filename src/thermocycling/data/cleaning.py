import pandas as pd

def apply_manual_trim(df: pd.DataFrame, gas: str) -> pd.DataFrame:
    """
    Apply one-off manual cropping you discovered empirically in notebooks.

    Currently a no-op; later you can paste the slicing logic you used
    (e.g. drop first 15 cycles of H2S, etc.).
    """
    if gas == 'H2S':
        df = df[15*402:]
    elif gas == 'NO2':
        df = df[:-400]
    elif gas == 'NO2_2':
        df = df[:5*120*402]
    else:
        pass
    switch_mask = (df['MFC_target'] == 0) & (df['MFC_target'].shift() == 25)
    incremental_values = switch_mask.cumsum()
    return df.assign(meas_cycle = incremental_values)
