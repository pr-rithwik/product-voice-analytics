# Gradio app — product voice analytics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import duckdb
import gradio as gr
import joblib
from sklearn.pipeline import Pipeline
from huggingface_hub import HfApi, hf_hub_download

from src.utils import stream_reviews_for_asin
from src.pipeline.preprocess import clean_text
from src.pipeline.sentiment import load_model, predict_tfidf, predict_distilbert
from src.intelligence.clustering import embed_reviews, cluster_reviews, build_topic_reviews
from src.intelligence.summarizer import generate_bullets
from src.config import (
    RAW_REVIEWS_PATH,
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

    DISTILBERT_CONFIG_PATH = os.path.join(DISTILBERT_PATH, 'config.json')
    if not os.path.exists(DISTILBERT_CONFIG_PATH):
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
        hf_hub_download(
            repo_id=HF_REPO,
            filename='product_lookup.csv',
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )


download_artifacts()


# load models
vectorizer     = joblib.load(str(TFIDF_VECTORIZER_PATH))
lr_model       = joblib.load(str(LR_MODEL_PATH))
tfidf_pipeline = Pipeline([('tfidf', vectorizer), ('lr', lr_model)])

distilbert_model, distilbert_tokenizer = load_model('distilbert')

# load cache
with open(str(DEMO_CACHE_PATH)) as f:
    DEMO_CACHE = json.load(f)

PRODUCT_NAMES    = {v['name']: k for k, v in DEMO_CACHE.items()}
DROPDOWN_CHOICES = ['-- Select a demo product --'] + list(PRODUCT_NAMES.keys()) + ['Custom Search']


def search_products(query, limit=20):
    if not query or len(query.strip()) < 2:
        return []
    result = duckdb.query(f"""
        SELECT asin, title
        FROM '{str(PRODUCT_LOOKUP_PATH)}'
        WHERE title ILIKE '%{query.strip()}%'
        LIMIT {limit}
    """).df()
    return result['title'].tolist()


def resolve_asin(title):
    result = duckdb.query(f"""
        SELECT asin
        FROM '{str(PRODUCT_LOOKUP_PATH)}'
        WHERE title = '{title.replace("'", "''")}'
        LIMIT 1
    """).df()
    if result.empty:
        return ''
    return result['asin'].iloc[0]


def format_results(total, breakdown, praise, complaints, source, model_used='TF-IDF + LR'):
    sentiment_summary = (
        f"Source: {source}\n"
        f"Model: {model_used}\n"
        f"Total reviews analysed: {total:,}\n\n"
        f"✅ Positive: {breakdown['positive']}%\n"
        f"😐 Neutral:  {breakdown['neutral']}%\n"
        f"❌ Negative: {breakdown['negative']}%"
    )
    praise_text    = '\n'.join([f'{i}. {b}' for i, b in enumerate(praise, 1)])
    complaint_text = '\n'.join([f'{i}. {b}' for i, b in enumerate(complaints, 1)])
    return sentiment_summary, praise_text, complaint_text


def analyse(product_selection, product_search, model_choice):

    # cached demo product
    if product_selection and product_selection not in ['-- Select a demo product --', 'Custom Search']:
        asin   = PRODUCT_NAMES[product_selection]
        result = DEMO_CACHE[asin]
        sentiment_summary, praise_text, complaint_text = format_results(
            result['total'],
            result['breakdown'],
            result['praise'],
            result['complaints'],
            source='pre-computed cache',
            model_used='TF-IDF + LR (pre-computed)'
        )
        return f'✅ Loaded: {product_selection}', sentiment_summary, praise_text, complaint_text

    # live pipeline via product search
    if product_selection == 'Custom Search' or product_search:
        asin = resolve_asin(product_search)
        if not asin:
            return 'Product not found. Try a different search term.', '', '', ''

        reviews = stream_reviews_for_asin(str(RAW_REVIEWS_PATH), asin)
        if not reviews:
            return f'No reviews found for ASIN {asin}.', '', '', ''

        clean_reviews = [clean_text(r) for r in reviews]
        clean_reviews = [r for r in clean_reviews if r]

        if model_choice == 'DistilBERT':
            labels = predict_distilbert(distilbert_model, distilbert_tokenizer, clean_reviews)
        else:
            labels = predict_tfidf(tfidf_pipeline, clean_reviews)

        total     = len(labels)
        breakdown = {
            'positive': round(labels.count('positive') / total * 100, 1),
            'neutral':  round(labels.count('neutral')  / total * 100, 1),
            'negative': round(labels.count('negative') / total * 100, 1),
        }

        embeddings    = embed_reviews(reviews)
        topics, _     = cluster_reviews(reviews, embeddings)
        topic_reviews = build_topic_reviews(reviews, topics)
        praise, complaints = generate_bullets(topic_reviews)

        sentiment_summary, praise_text, complaint_text = format_results(
            total, breakdown, praise, complaints,
            source='live analysis',
            model_used=model_choice
        )
        return f'✅ Done: {asin} ({total:,} reviews)', sentiment_summary, praise_text, complaint_text

    return 'Please select a demo product or search for one.', '', '', ''


with gr.Blocks(title='Product Voice Analytics') as demo:
    gr.Markdown('# 🔍 Product Voice Analytics')
    gr.Markdown('Select a demo product for instant results, or search any Amazon Electronics product for live analysis.')

    with gr.Row():
        product_dropdown = gr.Dropdown(
            choices=DROPDOWN_CHOICES,
            label='Demo Products',
            value=DROPDOWN_CHOICES[1]
        )
        model_choice = gr.Radio(
            ['TF-IDF + LR', 'DistilBERT'],
            label='Sentiment Model',
            value='TF-IDF + LR'
        )

    with gr.Accordion('Advanced — Analyse any product (may take several minutes)', open=False):
        product_search = gr.Dropdown(
            choices=[],
            label='Search Product by Name',
            allow_custom_value=True,
            filterable=True
        )
        gr.Markdown('⚠️ Streams the full dataset. Best used with TF-IDF + LR for speed.')

    analyse_btn   = gr.Button('Analyse', variant='primary')
    status_out    = gr.Textbox(label='Status',              interactive=False)
    sentiment_out = gr.Textbox(label='Sentiment Breakdown', interactive=False, lines=6)

    with gr.Row():
        praise_out    = gr.Textbox(label='✅ Top Praise Themes',    interactive=False, lines=8)
        complaint_out = gr.Textbox(label='⚠️ Top Complaint Themes', interactive=False, lines=8)

    product_search.change(
        fn=search_products,
        inputs=product_search,
        outputs=product_search
    )

    analyse_btn.click(
        fn=analyse,
        inputs=[product_dropdown, product_search, model_choice],
        outputs=[status_out, sentiment_out, praise_out, complaint_out]
    )

if __name__ == '__main__':
    demo.launch(share=True)