# Central config for all project-wide constants.
# Import from here instead of hardcoding values in individual modules.
import os
from pathlib import Path

from constants import DATA_PROCESSED_DIR
ROOT = Path(__file__).resolve().parent.parent

# data paths
RAW_REVIEWS_PATH = Path(os.environ.get('RAW_REVIEWS_PATH', str(ROOT / 'data/raw/Electronics.json.gz')))
RAW_METADATA_PATH = Path(os.environ.get('RAW_METADATA_PATH', str(ROOT / 'data/raw/meta_Electronics.json.gz')))
PROCESSED_SAMPLE_PATH = Path(os.environ.get('PROCESSED_SAMPLE_PATH', str(DATA_PROCESSED_DIR / 'sample_100k.csv')))

# sampling
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

CLASS_PROPORTIONS = {
    0: 0.20,  # negative (1-2 stars)
    1: 0.10,  # neutral  (3 stars)
    2: 0.70,  # positive (4-5 stars)
}

LABEL_MAP = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}

# train/test split
TEST_SIZE = 0.2


# distilbert
DISTILBERT_MAX_LEN = 128
DISTILBERT_BATCH_SIZE = 64


# topic intelligence
EMBED_MODEL = "all-MiniLM-L6-v2"
N_TOPICS = 10
N_BULLETS = 5