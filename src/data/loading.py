from pathlib import Path
import pandas as pd
from .paths import RAW_DATA_DIR, GAS_FILE_MAP

def load_raw_gas(gas: str, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load raw thermocycling data for a single gas.

    Parameters
    ----------
    gas : str
        One of the keys in GAS_FILE_MAP, e.g. "NO2", "H2S", "Acet".
    data_dir : Path, optional
        Base directory with raw data. Defaults to data/raw inside the repo.

    Returns
    -------
    pd.DataFrame
        Raw measurement dataframe as stored on disk.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    if gas not in GAS_FILE_MAP:
        raise ValueError(f"Unknown gas {gas!r}. Known: {list(GAS_FILE_MAP)}")

    path = data_dir / GAS_FILE_MAP[gas]
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Use read_parquet by default; change to read_csv if needed
    return pd.read_parquet(path)