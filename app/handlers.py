# app/handlers.py — core analysis logic
from src.utils import get_reviews_for_asin
from src.pipeline.preprocess import clean_text
from src.pipeline.sentiment import predict_tfidf, predict_distilbert
from src.intelligence.clustering import embed_reviews, cluster_reviews, build_topic_reviews
from src.intelligence.summarizer import generate_bullets

from app import resolve_asin


def is_valid_bullet(bullet: str) -> bool:
    skip_prefixes = (
        "I cannot",
        "I don't",
        "I appreciate",
        "I understand",
        "I can't",
        "I need",
        "If you",
        "Note that",
        "Please provide",
    )
    skip_substrings = (
        "these reviews",
        "no complaint",
        "all positive",
        "only positive",
    )
    b = bullet.strip()
    if any(b.startswith(p) for p in skip_prefixes):
        return False
    if any(s.lower() in b.lower() for s in skip_substrings):
        return False
    return True


def format_results(total, breakdown, praise, complaints, source, model_used='TF-IDF + LR'):
    sentiment_summary = (
        f"Source: {source}\n"
        f"Model: {model_used}\n"
        f"Total reviews analysed: {total:,}\n\n"
        f"✅ Positive: {breakdown['positive']}%\n"
        f"😐 Neutral:  {breakdown['neutral']}%\n"
        f"❌ Negative: {breakdown['negative']}%"
    )
    praise = [b for b in praise if is_valid_bullet(b)]
    complaints = [b for b in complaints if is_valid_bullet(b)]
    praise_text = '\n'.join([f'{i}. {b}' for i, b in enumerate(praise, 1)]) or 'No strong praise themes identified.'
    complaint_text = '\n'.join([f'{i}. {b}' for i, b in enumerate(complaints, 1)]) or 'No strong complaint themes identified.'
    return sentiment_summary, praise_text, complaint_text


def analyse(product_selection, search_result, model_choice, tfidf_pipeline, distilbert_model, distilbert_tokenizer, demo_cache, product_names):

    # live pipeline takes priority if a search result is selected
    if search_result:
        asin = resolve_asin(search_result)
        if not asin:
            return 'Product not found. Try a different search term.', '', '', ''

        reviews = get_reviews_for_asin(asin)
        if not reviews:
            return f'No reviews found for ASIN {asin}.', '', '', ''

        clean_reviews = [clean_text(r) for r in reviews]
        clean_reviews = [r for r in clean_reviews if r]

        if model_choice == 'DistilBERT':
            labels = predict_distilbert(distilbert_model, distilbert_tokenizer, clean_reviews)
        else:
            labels = predict_tfidf(tfidf_pipeline, clean_reviews)

        total = len(labels)
        breakdown = {
            'positive': round(labels.count('positive') / total * 100, 1),
            'neutral':  round(labels.count('neutral')  / total * 100, 1),
            'negative': round(labels.count('negative') / total * 100, 1),
        }

        # BERTopic needs enough reviews to cluster
        if len(reviews) < 50:
            return f'Not enough reviews for topic analysis (found {len(reviews)}).', '', '', ''

        # split by sentiment before clustering so Claude always summarizes the right group
        positive_reviews = [r for r, l in zip(reviews, labels) if l == 'positive']
        negative_reviews = [r for r, l in zip(reviews, labels) if l == 'negative']

        praise, complaints = [], []

        if len(positive_reviews) >= 10:
            pos_embeddings = embed_reviews(positive_reviews)
            pos_topics, _ = cluster_reviews(positive_reviews, pos_embeddings)
            pos_topic_reviews = build_topic_reviews(positive_reviews, pos_topics)
            praise, _ = generate_bullets(pos_topic_reviews)

        if len(negative_reviews) >= 10:
            neg_embeddings = embed_reviews(negative_reviews)
            neg_topics, _ = cluster_reviews(negative_reviews, neg_embeddings)
            neg_topic_reviews = build_topic_reviews(negative_reviews, neg_topics)
            _, complaints = generate_bullets(neg_topic_reviews)

        sentiment_summary, praise_text, complaint_text = format_results(
            total, breakdown, praise, complaints,
            source='live analysis',
            model_used=model_choice
        )
        return f'✅ Done: {asin} ({total:,} reviews)', sentiment_summary, praise_text, complaint_text

    # cached demo product
    if product_selection and product_selection not in ['-- Select a demo product --', 'Custom Search']:
        asin = product_names[product_selection]
        result = demo_cache[asin]
        sentiment_summary, praise_text, complaint_text = format_results(
            result['total'],
            result['breakdown'],
            result['praise'],
            result['complaints'],
            source='pre-computed cache',
            model_used='TF-IDF + LR (pre-computed)'
        )
        return f'✅ Loaded: {product_selection}', sentiment_summary, praise_text, complaint_text

    return 'Please select a demo product or search for one.', '', '', ''