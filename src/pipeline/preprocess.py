import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from src.config import PROCESSED_SAMPLE_PATH

nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', ' ', text)


def remove_special_chars(text: str) -> str:
    """Remove punctuation and non-alphabetic characters, collapse whitespace."""
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from whitespace-tokenized text."""
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single review string.
    Returns empty string for null or non-string input.
    """
    is_valid = isinstance(text, str) and bool(text.strip())
    if not is_valid:
        return ''

    text = text.lower()
    text = strip_html(text)
    text = remove_special_chars(text)
    text = remove_stopwords(text)
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text to the reviewText column and store result in clean_text.
    Returns a copy of the DataFrame with clean_text populated.

    Args:
        df: DataFrame with a reviewText column

    Returns:
        DataFrame with clean_text column filled
    """
    df = df.copy()
    df['clean_text'] = df['reviewText'].apply(clean_text)
    return df


if __name__ == '__main__':
    print('Loading sample...')
    df = pd.read_csv(PROCESSED_SAMPLE_PATH)

    print('Cleaning...')
    df = preprocess(df)

    print('Saving...')
    df.to_csv(PROCESSED_SAMPLE_PATH, index=False)

    print('Done.')