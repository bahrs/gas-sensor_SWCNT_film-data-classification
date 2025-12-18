from pathlib import Path
import pandas as pd  # type: ignore[import]
from .paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RAW_FILE_MAP, PROCESSED_FILE_MAP

def load_gas_data(gas: str, raw: bool = False) -> pd.DataFrame:
    """
    Load raw thermocycling data for a single gas.

    Parameters
    ----------
    gas : str
        One of the keys in GAS_FILE_MAP, e.g. "NO2", "H2S", "Acet" or "NO2_2" - a set of NO2 measurements recorded 2 months later.
    data_dir : Path, optional
        Base directory with raw data. Defaults to data/raw inside the repo.

    Returns
    -------
    pd.DataFrame
        Raw measurement dataframe as stored on disk.
    """

    if gas not in RAW_FILE_MAP.keys():
        raise ValueError(f"Unknown gas {gas!r}. Known: {list(RAW_FILE_MAP.keys())}")

    if raw:
        path = RAW_DATA_DIR / RAW_FILE_MAP[gas]
    else:
        path = PROCESSED_DATA_DIR / PROCESSED_FILE_MAP[gas]
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_parquet(path)