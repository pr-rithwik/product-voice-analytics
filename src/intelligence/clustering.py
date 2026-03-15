# Embeds reviews and clusters them using BERTopic.
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from src.config import EMBED_MODEL, N_TOPICS


def embed_reviews(reviews: list[str]) -> any:
    embedder = SentenceTransformer(EMBED_MODEL)
    return embedder.encode(reviews, show_progress_bar=True)


def cluster_reviews(reviews: list[str], embeddings) -> tuple[list[int], BERTopic]:
    n_topics = min(N_TOPICS, len(reviews) // 5)
    n_topics = max(n_topics, 2)

    topic_model = BERTopic(nr_topics=n_topics, language='english', calculate_probabilities=False, min_topic_size=3)
    topics, _ = topic_model.fit_transform(reviews, embeddings)
    return topics, topic_model


def get_topic_reviews(reviews: list[str], topics: list[int], topic_id: int, n: int = 10) -> list[str]:
    indices = [i for i, t in enumerate(topics) if t == topic_id]
    return [reviews[i] for i in indices[:n]]


def get_valid_topics(topics: list[int]) -> list[int]:
    return [t for t in set(topics) if t != -1]


def build_topic_reviews(reviews: list[str], topics: list[int]) -> dict[int, list[str]]:
    valid_ids = get_valid_topics(topics)
    return {tid: get_topic_reviews(reviews, topics, tid) for tid in valid_ids}