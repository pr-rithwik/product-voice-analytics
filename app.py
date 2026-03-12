# Gradio app — product voice analytics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as gr
import joblib
from sklearn.pipeline import Pipeline
from huggingface_hub import HfApi, hf_hub_download

from src.utils import stream_reviews_for_asin
from src.pipeline.preprocess import clean_text
from src.pipeline.sentiment import load_model, predict_tfidf, predict_distilbert
from src.intelligence.clustering import embed_reviews, cluster_reviews, build_topic_reviews
from src.intelligence.summarizer import generate_bullets
from src.config import (
    RAW_REVIEWS_PATH, TFIDF_VECTORIZER_PATH, LR_MODEL_PATH,
    DISTILBERT_PATH
)

HF_REPO = 'rithweek/product-voice-analytics-models'

api = HfApi()
os.makedirs('models', exist_ok=True)
os.makedirs(str(DISTILBERT_PATH), exist_ok=True)

# download pkl files
if not os.path.exists(str(TFIDF_VECTORIZER_PATH)):
    hf_hub_download(repo_id=HF_REPO, filename='tfidf_vectorizer.pkl', local_dir='models', local_dir_use_symlinks=False)

if not os.path.exists(str(LR_MODEL_PATH)):
    hf_hub_download(repo_id=HF_REPO, filename='lr_model.pkl', local_dir='models', local_dir_use_symlinks=False)

# download distilbert files dynamically
all_files        = api.list_repo_files(repo_id=HF_REPO, repo_type='model')
distilbert_files = [f for f in all_files if f.startswith('distilbert/')]

for filepath in distilbert_files:
    local_path = os.path.join(str(DISTILBERT_PATH), os.path.basename(filepath))
    if not os.path.exists(local_path):
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filepath,
            local_dir='models',
            local_dir_use_symlinks=False
        )

#debug
for root, dirs, files in os.walk('models'):
    for file in files:
        print(os.path.join(root, file))

# load models once at startup
vectorizer   = joblib.load(TFIDF_VECTORIZER_PATH)
lr_model     = joblib.load(LR_MODEL_PATH)
tfidf_pipeline = Pipeline([('tfidf', vectorizer), ('lr', lr_model)])

distilbert_model, distilbert_tokenizer = load_model('distilbert')


def analyse(asin: str, model_choice: str) -> tuple:
    asin = asin.strip().upper()

    if not asin:
        return 'Please enter an ASIN.', '', '', ''

    # stream reviews
    reviews = stream_reviews_for_asin(str(RAW_REVIEWS_PATH), asin)
    if not reviews:
        return f'No reviews found for ASIN {asin}.', '', '', ''

    # preprocess
    clean_reviews = [clean_text(r) for r in reviews]
    clean_reviews = [r for r in clean_reviews if r]

    # sentiment
    if model_choice == 'DistilBERT':
        labels = predict_distilbert(distilbert_model, distilbert_tokenizer, clean_reviews)
    else:
        labels = predict_tfidf(tfidf_pipeline, clean_reviews)

    total = len(labels)
    pos   = round(labels.count('positive') / total * 100, 1)
    neu   = round(labels.count('neutral')  / total * 100, 1)
    neg   = round(labels.count('negative') / total * 100, 1)

    sentiment_summary = (
        f"Total reviews analysed: {total:,}\n\n"
        f"✅ Positive: {pos}%\n"
        f"😐 Neutral:  {neu}%\n"
        f"❌ Negative: {neg}%"
    )

    # topic intelligence
    embeddings    = embed_reviews(reviews)
    topics, _     = cluster_reviews(reviews, embeddings)
    topic_reviews = build_topic_reviews(reviews, topics)

    praise_bullets, complaint_bullets = generate_bullets(topic_reviews)

    praise_text    = '\n'.join([f'{i}. {b}' for i, b in enumerate(praise_bullets, 1)])
    complaint_text = '\n'.join([f'{i}. {b}' for i, b in enumerate(complaint_bullets, 1)])

    return f'Found {total:,} reviews for {asin}', sentiment_summary, praise_text, complaint_text


with gr.Blocks(title='Product Voice Analytics') as demo:
    gr.Markdown('# 🔍 Product Voice Analytics')
    gr.Markdown('Enter an Amazon Electronics ASIN to get sentiment breakdown and key themes from customer reviews.')

    with gr.Row():
        asin_input    = gr.Textbox(label='Product ASIN', placeholder='e.g. B07XJ8C8F5')
        model_choice  = gr.Radio(['TF-IDF + LR', 'DistilBERT'], label='Sentiment Model', value='TF-IDF + LR')

    analyse_btn = gr.Button('Analyse', variant='primary')

    status_out    = gr.Textbox(label='Status', interactive=False)
    sentiment_out = gr.Textbox(label='Sentiment Breakdown', interactive=False, lines=5)

    with gr.Row():
        praise_out    = gr.Textbox(label='✅ Top Praise Themes',    interactive=False, lines=7)
        complaint_out = gr.Textbox(label='⚠️ Top Complaint Themes', interactive=False, lines=7)

    analyse_btn.click(
        fn=analyse,
        inputs=[asin_input, model_choice],
        outputs=[status_out, sentiment_out, praise_out, complaint_out]
    )


if __name__ == '__main__':
    demo.launch()