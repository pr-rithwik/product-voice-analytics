# app.py — entry point for HuggingFace Space and local runs
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import joblib
from sklearn.pipeline import Pipeline

from src.pipeline.sentiment import load_model
from src.config import TFIDF_VECTORIZER_PATH, LR_MODEL_PATH, DEMO_CACHE_PATH

from app import download_artifacts, build_ui

# download all artifacts from HuggingFace
download_artifacts()

# load models
vectorizer     = joblib.load(str(TFIDF_VECTORIZER_PATH))
lr_model       = joblib.load(str(LR_MODEL_PATH))
tfidf_pipeline = Pipeline([('tfidf', vectorizer), ('lr', lr_model)])

distilbert_model, distilbert_tokenizer = load_model('distilbert')

# load cache
with open(str(DEMO_CACHE_PATH)) as f:
    DEMO_CACHE = json.load(f)

product_names = {v['name']: k for k, v in DEMO_CACHE.items()}

# build and launch UI
demo = build_ui(
    tfidf_pipeline=tfidf_pipeline,
    distilbert_model=distilbert_model,
    distilbert_tokenizer=distilbert_tokenizer,
    demo_cache=DEMO_CACHE,
    product_names=product_names
)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)