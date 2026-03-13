# Streamlit app — product voice analytics
# Run: streamlit run app/main.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.utils import stream_reviews_for_asin
from src.pipeline.preprocess import clean_text
from src.pipeline.sentiment import load_model, predict_tfidf, predict_distilbert
from src.intelligence.clustering import embed_reviews, cluster_reviews, build_topic_reviews
from src.intelligence.summarizer import generate_bullets
from src.config import RAW_REVIEWS_PATH, TFIDF_VECTORIZER_PATH, LR_MODEL_PATH, DISTILBERT_PATH
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# page config
st.set_page_config(
    page_title='Product Voice Analytics',
    page_icon='🔍',
    layout='wide'
)


@st.cache_resource
def load_tfidf():
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    lr_model   = joblib.load(LR_MODEL_PATH)
    return Pipeline([('tfidf', vectorizer), ('lr', lr_model)])


@st.cache_resource
def load_distilbert():
    return load_model('distilbert')


def sentiment_breakdown(labels: list[str]) -> dict:
    total = len(labels)
    counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    for label in labels:
        counts[label] += 1
    return {k: round(v / total * 100, 1) for k, v in counts.items()}


# header
st.title('🔍 Product Voice Analytics')
st.markdown('Enter an Amazon Electronics ASIN to get sentiment breakdown and key themes from customer reviews.')

# sidebar — model selection
st.sidebar.header('Settings')
model_choice = st.sidebar.radio('Sentiment Model', ['TF-IDF + LR (fast)', 'DistilBERT (accurate)'])
use_distilbert = model_choice == 'DistilBERT (accurate)'

# input
asin = st.text_input('Product ASIN', placeholder='e.g. B07XJ8C8F5').strip().upper()

if st.button('Analyse', type='primary') and asin:

    # 1. stream reviews
    with st.spinner(f'Streaming reviews for {asin}...'):
        reviews = stream_reviews_for_asin(str(RAW_REVIEWS_PATH), asin)

    if not reviews:
        st.error(f'No reviews found for ASIN {asin}. Please check the ASIN and try again.')
        st.stop()

    st.success(f'Found {len(reviews):,} reviews')

    # 2. clean text
    with st.spinner('Preprocessing...'):
        clean_reviews = [clean_text(r) for r in reviews]
        clean_reviews = [r for r in clean_reviews if r]

    # 3. sentiment
    with st.spinner('Running sentiment analysis...'):
        if use_distilbert:
            model, tokenizer = load_distilbert()
            labels = predict_distilbert(model, tokenizer, clean_reviews)
        else:
            pipeline = load_tfidf()
            labels = predict_tfidf(pipeline, clean_reviews)

    breakdown = sentiment_breakdown(labels)

    # sentiment display
    st.subheader('📊 Sentiment Breakdown')
    col1, col2, col3 = st.columns(3)
    col1.metric('✅ Positive', f"{breakdown['positive']}%")
    col2.metric('😐 Neutral',  f"{breakdown['neutral']}%")
    col3.metric('❌ Negative', f"{breakdown['negative']}%")

    sentiment_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Percentage': [breakdown['positive'], breakdown['neutral'], breakdown['negative']]
    })
    st.bar_chart(sentiment_df.set_index('Sentiment'))

    # 4. topic intelligence
    st.subheader('💡 Topic Intelligence')

    with st.spinner('Embedding and clustering reviews (this may take a minute)...'):
        embeddings   = embed_reviews(reviews)
        topics, _    = cluster_reviews(reviews, embeddings)
        topic_reviews = build_topic_reviews(reviews, topics)

    with st.spinner('Summarizing themes with Claude...'):
        praise_bullets, complaint_bullets = generate_bullets(topic_reviews)

    col_praise, col_complaint = st.columns(2)

    with col_praise:
        st.markdown('### ✅ Top Praise Themes')
        for i, bullet in enumerate(praise_bullets, 1):
            st.markdown(f'**{i}.** {bullet}')

    with col_complaint:
        st.markdown('### ⚠️ Top Complaint Themes')
        for i, bullet in enumerate(complaint_bullets, 1):
            st.markdown(f'**{i}.** {bullet}')

    # raw review sample
    with st.expander('View sample reviews'):
        for review in reviews[:5]:
            st.markdown(f'> {review[:300]}...')
            st.divider()