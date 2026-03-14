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


def stream_reviews_for_asin(raw_path, asin, max_reviews=2000):
    reviews = []
    raw_path = str(raw_path)
    
    opener = gzip.open if raw_path.endswith('.gz') else open
    
    with opener(raw_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get('asin') == asin:
                    text = record.get('reviewText', '').strip()
                    if text:
                        reviews.append(text)
                        if len(reviews) >= max_reviews:
                            break
            except:
                continue
    return reviews