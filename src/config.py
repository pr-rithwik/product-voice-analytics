from pathlib import Path

# repo root — all paths are relative to this
ROOT = Path(__file__).resolve().parent.parent


# data paths
RAW_REVIEWS_PATH = ROOT / "data/raw/Electronics.json.gz"
RAW_METADATA_PATH = ROOT / "data/raw/meta_Electronics.json.gz"
PROCESSED_SAMPLE_PATH = ROOT / "data/processed/sample_100k.csv"


# sampling
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

CLASS_PROPORTIONS = {
    0: 0.20,  # negative (1-2 stars)
    1: 0.10,  # neutral  (3 stars)
    2: 0.70,  # positive (4-5 stars)
}


# train/test split
TEST_SIZE = 0.2