# app/artifacts.py — downloads models and data from HuggingFace at startup
import os
from huggingface_hub import HfApi, hf_hub_download

from constants import (
    HF_MODEL_REPO, HF_DATASET_REPO, HF_REPO_TYPE_DATASET, 
    HF_REPO_TYPE_MODEL, APP_CACHE_DIR, DATA_PROCESSED_DIR,
    TFIDF_VECTORIZER_PATH, LR_MODEL_PATH, DISTILBERT_DIR,
    PARQUET_PATH, TFIDF_VECTORIZER_FILE, LR_MODEL_FILE,
    MODELS_DIR, DISTILBERT, PARQUET_FILE, PRODUCT_LOOKUP_FILE,
    DISTILBERT_CONFIG_PATH, PRODUCT_LOOKUP_PATH
)


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(APP_CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


def download_artifacts():
    api = HfApi()

    # TF-IDF downloader
    if not os.path.exists(str(TFIDF_VECTORIZER_PATH)):
        hf_hub_download(
            repo_id=HF_MODEL_REPO, 
            filename=TFIDF_VECTORIZER_FILE, 
            local_dir=str(MODELS_DIR), 
            local_dir_use_symlinks=False
        )

    # LR downloader
    if not os.path.exists(str(LR_MODEL_PATH)):
        hf_hub_download(
            repo_id=HF_MODEL_REPO, 
            filename=LR_MODEL_FILE, 
            local_dir=str(MODELS_DIR), 
            local_dir_use_symlinks=False
        )

    # DISTILBERT downloader
    if not os.path.exists(DISTILBERT_CONFIG_PATH):
        all_files = api.list_repo_files(repo_id=HF_MODEL_REPO, repo_type=HF_REPO_TYPE_MODEL)
        distilbert_files = [f for f in all_files if f.startswith(f'{DISTILBERT}/')]
        for filepath in distilbert_files:
            local_path = os.path.join(DISTILBERT_DIR, os.path.basename(filepath))
            if not os.path.exists(local_path):
                hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    filename=filepath,
                    local_dir=str(MODELS_DIR),
                    local_dir_use_symlinks=False
                )

    # Product Lookup downloader
    if not os.path.exists(str(PRODUCT_LOOKUP_PATH)):
        hf_hub_download(
            repo_id=HF_DATASET_REPO, 
            filename=PRODUCT_LOOKUP_FILE, 
            local_dir=str(DATA_PROCESSED_DIR),
            repo_type=HF_REPO_TYPE_DATASET, 
            local_dir_use_symlinks=False
        )

    if not os.path.exists(str(PARQUET_PATH)):
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=PARQUET_FILE,
            local_dir=str(DATA_PROCESSED_DIR),
            repo_type=HF_REPO_TYPE_DATASET,
            local_dir_use_symlinks=False
        )