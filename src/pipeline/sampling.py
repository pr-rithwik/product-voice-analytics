import gzip
import json
import os
import random
import pandas as pd
from src.config import (
    RAW_REVIEWS_PATH,
    PROCESSED_SAMPLE_PATH,
    SAMPLE_SIZE,
    RANDOM_SEED,
    CLASS_PROPORTIONS,
)


def rating_to_label(overall: float) -> int:
    """Convert star rating to 3-class sentiment label."""
    if overall <= 2.0:
        return 0
    elif overall == 3.0:
        return 1
    else:
        return 2


def reservoir_sample(raw_path: str, total_n: int, proportions: dict, 
                     seed: int = None) -> list[dict]:
    """
    Single-pass stratified reservoir sampling over a gzipped JSON Lines file.
    Memory usage stays bounded at total_n records regardless of file size.

    For each class, maintains a reservoir of size n_class.
    When the reservoir is full, each new record replaces a random existing
    one with decreasing probability — this guarantees uniform random sampling.

    Args:
        raw_path:    path to Electronics.json.gz
        total_n:     total records to sample
        proportions: fraction of total_n per class label
        seed:        random seed

    Returns:
        list of dicts with keys: reviewText, overall, label, clean_text
    """
    seed = RANDOM_SEED if seed is None else seed
    random.seed(seed)

    targets = {label: int(total_n * frac) for label, frac in proportions.items()}
    reservoirs = {label: [] for label in proportions}
    counts = {label: 0 for label in proportions}

    with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            has_text = bool(record.get('reviewText', '').strip())
            has_rating = record.get('overall') is not None
            if not (has_text and has_rating):
                continue

            label = rating_to_label(float(record['overall']))
            counts[label] += 1
            reservoir = reservoirs[label]
            target = targets[label]

            if len(reservoir) < target:
                reservoir.append({
                    'reviewText': record['reviewText'],
                    'overall':    record['overall'],
                    'label':      label,
                    'clean_text': '',
                })
            else:
                j = random.randint(0, counts[label] - 1)
                if j < target:
                    reservoir[j] = {
                        'reviewText': record['reviewText'],
                        'overall':    record['overall'],
                        'label':      label,
                        'clean_text': '',
                    }

    for label, reservoir in reservoirs.items():
        print(f'Label {label}: {len(reservoir)} records sampled')

    all_records = []
    for label, reservoir in reservoirs.items():
        all_records += reservoir

    return all_records


def build_dataframe(records: list[dict], seed: int) -> pd.DataFrame:
    """Shuffle records and return as a clean DataFrame."""
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed)
    df = df.reset_index(drop=True)
    print(f'Final sample shape: {df.shape}')
    print(df['label'].value_counts().sort_index())
    return df


def save_sample(df: pd.DataFrame, output_path: str) -> None:
    """Save sample DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    print('Starting reservoir sampling...')
    records = reservoir_sample(RAW_REVIEWS_PATH, SAMPLE_SIZE, CLASS_PROPORTIONS, RANDOM_SEED)

    print('\nBuilding dataframe...')
    df = build_dataframe(records, RANDOM_SEED)

    print('\nSaving...')
    save_sample(df, PROCESSED_SAMPLE_PATH)