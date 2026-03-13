# Central config for all project-wide constants.
# Import from here instead of hardcoding values in individual modules.
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# data paths
RAW_REVIEWS_PATH = ROOT / "data/raw/Electronics.json.gz"
RAW_METADATA_PATH = ROOT / "data/raw/meta_Electronics.json.gz"
PROCESSED_SAMPLE_PATH = ROOT / "data/processed/sample_100k.csv"


# model paths
MODELS_DIR = Path(os.environ.get('MODELS_DIR', str(ROOT / 'models')))
DISTILBERT_PATH = Path(os.environ.get('DISTILBERT_PATH', str(MODELS_DIR / 'distilbert')))
TFIDF_VECTORIZER_PATH = Path(os.environ.get('TFIDF_VECTORIZER_PATH', str(MODELS_DIR / 'tfidf_vectorizer.pkl')))
LR_MODEL_PATH = Path(os.environ.get('LR_MODEL_PATH', str(MODELS_DIR / 'lr_model.pkl')))
# BEST_MODEL_REF_PATH = Path(os.environ.get('BEST_MODEL_REF_PATH', str(MODELS_DIR / 'best_model_ref.txt')))
DEMO_CACHE_PATH = Path(os.environ.get('DEMO_CACHE_PATH', str(MODELS_DIR / 'demo_cache.json')))


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


# distilbert
DISTILBERT_MAX_LEN = 128
DISTILBERT_BATCH_SIZE = 64


# topic intelligence
EMBED_MODEL = "all-MiniLM-L6-v2"
N_TOPICS = 10
N_BULLETS = 5