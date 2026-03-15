# Summarizes review clusters into plain-English bullets via Claude API.

import anthropic
from src.config import N_BULLETS

client = anthropic.Anthropic()


def summarize_topic(reviews: list[str], sentiment: str) -> str:
    reviews_block = '\n'.join(reviews)

    prompt = f"""You are analyzing Amazon Electronics product reviews.

The following reviews share a common {sentiment} theme:

{reviews_block}

Summarize the core {sentiment} theme into ONE concise bullet point (max 20 words).
Start with an action verb.
Return only the bullet text, nothing else."""

    response = client.messages.create(
        model = 'claude-haiku-4-5-20251001',
        max_tokens = 60,
        messages = [{'role': 'user', 'content': prompt}]
    )

    return response.content[0].text.strip()


def generate_bullets(topic_reviews: dict[int, list[str]]) -> tuple[list[str], list[str]]:
    sorted_topics = sorted(topic_reviews.keys(), key=lambda tid: len(topic_reviews[tid]), reverse=True)

    praise_ids = sorted_topics[:N_BULLETS]
    complaint_ids = sorted_topics[N_BULLETS:N_BULLETS * 2]

    praise_bullets = [summarize_topic(topic_reviews[tid], 'praise')    for tid in praise_ids]
    complaint_bullets = [summarize_topic(topic_reviews[tid], 'complaint') for tid in complaint_ids]

    return praise_bullets, complaint_bullets