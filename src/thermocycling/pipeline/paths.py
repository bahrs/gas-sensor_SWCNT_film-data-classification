from pathlib import Path

# Base project directory (assumes src/ is inside repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_FILE_MAP = {
    "NO2":  "NO2_15_02_protocol10-15-25.brotli",
    "H2S":  "H2S_17_02_protocol10-15-25.brotli",
    "Acet": "Acet_19_02_protocol10-15-25.brotli",
    "NO2_2": "NO2_16_04_protocol10-15-25.brotli",
}
PROCESSED_FILE_MAP = {
    "NO2": "NO2.parquet",
    "H2S": "H2S.parquet",
    "Acet": "Acet.parquet",
    "NO2_2": "NO2_2.parquet",
 }