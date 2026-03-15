# Shared utility functions used across the project.
import torch
import duckdb

from constants import PARQUET_PATH


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_reviews_for_asin(asin, max_reviews=2000):
    result = duckdb.query(f"""
        SELECT reviewText
        FROM '{str(PARQUET_PATH)}'
        WHERE asin = '{asin}'
        LIMIT {max_reviews}
    """).df()
    return result['reviewText'].tolist()
