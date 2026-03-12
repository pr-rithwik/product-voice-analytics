# Shared utility functions used across the project.

import gzip
import json
import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def stream_reviews_for_asin(raw_path: str, asin: str) -> list[str]:
    reviews = []
    with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record.get('asin') == asin:
                text = record.get('reviewText', '').strip()
                if text:
                    reviews.append(text)
    return reviews