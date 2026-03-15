import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get('MODELS_DIR', str(ROOT / 'models')))

# File AND folder names
PARQUET_FILE = "reviews.parquet"
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"
LR_MODEL_FILE = "lr_model.pkl"
DEMO_CACHE_FILE = "demo_cache.json"
PRODUCT_LOOKUP_FILE = "product_lookup.csv"
CONFIG_FILE = "config.json"

DISTILBERT = "distilbert"
APP_CACHE = "app/cache"
DATA_PROCESSED = "data/processed"

# HuggingFace remote
HF_MODEL_REPO = "rithweek/product-voice-analytics-models"
HF_DATASET_REPO = "rithweek/amazon-electronics-reviews-data"

HF_REPO_TYPE_MODEL = "model"
HF_REPO_TYPE_DATASET = "dataset"

# artifact names (used by app + pipeline)
APP_CACHE_DIR = Path(os.environ.get('APP_CACHE_DIR', str(ROOT / APP_CACHE)))
DATA_PROCESSED_DIR = Path(os.environ.get('DATA_PROCESSED_DIR', str(ROOT / DATA_PROCESSED)))

DISTILBERT_DIR = Path(os.environ.get('DISTILBERT_DIR', str(MODELS_DIR / DISTILBERT)))
DISTILBERT_CONFIG_PATH = Path(os.environ.get('DISTILBERT_CONFIG_PATH', str(DISTILBERT_DIR / CONFIG_FILE)))
TFIDF_VECTORIZER_PATH = Path(os.environ.get('TFIDF_VECTORIZER_PATH', str(MODELS_DIR / TFIDF_VECTORIZER_FILE)))
LR_MODEL_PATH = Path(os.environ.get('LR_MODEL_PATH', str(MODELS_DIR / LR_MODEL_FILE)))

DEMO_CACHE_PATH = Path(os.environ.get('DEMO_CACHE_PATH', str(APP_CACHE_DIR / DEMO_CACHE_FILE)))
PRODUCT_LOOKUP_PATH = Path(os.environ.get('PRODUCT_LOOKUP_PATH', str(DATA_PROCESSED_DIR / PRODUCT_LOOKUP_FILE)))
PARQUET_PATH = Path(os.environ.get('PARQUET_PATH', str(DATA_PROCESSED_DIR / PARQUET_FILE)))
