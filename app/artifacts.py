# app/artifacts.py — downloads models and data from HuggingFace at startup
import os
from huggingface_hub import HfApi, hf_hub_download

from src.config import (
    TFIDF_VECTORIZER_PATH,
    LR_MODEL_PATH,
    DISTILBERT_PATH,
    DEMO_CACHE_PATH,
    MODELS_DIR,
    PRODUCT_LOOKUP_PATH
)

HF_REPO = 'rithweek/product-voice-analytics-models'


def download_artifacts():
    api = HfApi()
    os.makedirs(DISTILBERT_PATH, exist_ok=True)

    if not os.path.exists(str(TFIDF_VECTORIZER_PATH)):
        hf_hub_download(repo_id=HF_REPO, filename='tfidf_vectorizer.pkl', local_dir=str(MODELS_DIR), local_dir_use_symlinks=False)

    if not os.path.exists(str(LR_MODEL_PATH)):
        hf_hub_download(repo_id=HF_REPO, filename='lr_model.pkl', local_dir=str(MODELS_DIR), local_dir_use_symlinks=False)

    distilbert_config = os.path.join(DISTILBERT_PATH, 'config.json')
    if not os.path.exists(distilbert_config):
        all_files        = api.list_repo_files(repo_id=HF_REPO, repo_type='model')
        distilbert_files = [f for f in all_files if f.startswith('distilbert/')]
        for filepath in distilbert_files:
            local_path = os.path.join(DISTILBERT_PATH, os.path.basename(filepath))
            if not os.path.exists(local_path):
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename=filepath,
                    local_dir=str(MODELS_DIR),
                    local_dir_use_symlinks=False
                )

    if not os.path.exists(str(DEMO_CACHE_PATH)):
        hf_hub_download(repo_id=HF_REPO, filename='demo_cache.json', local_dir=str(MODELS_DIR), local_dir_use_symlinks=False)

    if not os.path.exists(str(PRODUCT_LOOKUP_PATH)):
        hf_hub_download(repo_id=HF_REPO, filename='product_lookup.csv', local_dir=str(MODELS_DIR), local_dir_use_symlinks=False)