import torch
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.config import (
    TFIDF_PIPELINE_PATH,
    DISTILBERT_PATH,
    DISTILBERT_MAX_LEN,
    DISTILBERT_BATCH_SIZE,
)

LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}


def get_device() -> torch.device:
    """Return the best available device: MPS (Apple), CUDA (Windows/Linux), or CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_model(model_type: str) -> tuple:
    """
    Load and return the requested model.

    Args:
        model_type: 'tfidf' or 'distilbert'

    Returns:
        For tfidf:      (pipeline, None)
        For distilbert: (model, tokenizer)

    Raises:
        ValueError if model_type is not recognised
    """
    if model_type == 'tfidf':
        pipeline = joblib.load(TFIDF_PIPELINE_PATH)
        return pipeline, None

    elif model_type == 'distilbert':
        tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        model     = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
        model     = model.to(get_device())
        model.eval()
        return model, tokenizer

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'tfidf' or 'distilbert'.")


def predict_tfidf(pipeline, texts: list[str]) -> list[str]:
    """
    Run inference using the TF-IDF + LR pipeline.

    Args:
        pipeline: loaded sklearn pipeline
        texts:    list of cleaned review strings

    Returns:
        list of label strings ('negative', 'neutral', 'positive')
    """
    preds = pipeline.predict(texts)
    return [LABEL_MAP[p] for p in preds]


def predict_distilbert(model, tokenizer, texts: list[str]) -> list[str]:
    """
    Run batched inference using DistilBERT.

    Args:
        model:     loaded DistilBertForSequenceClassification
        tokenizer: matching tokenizer
        texts:     list of cleaned review strings

    Returns:
        list of label strings ('negative', 'neutral', 'positive')
    """
    device = get_device()
    all_preds = []

    for i in range(0, len(texts), DISTILBERT_BATCH_SIZE):
        batch_texts = texts[i: i + DISTILBERT_BATCH_SIZE]

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=DISTILBERT_MAX_LEN,
            return_tensors='pt'
        )

        input_ids      = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        all_preds.extend(preds)

    return [LABEL_MAP[p] for p in all_preds]


def predict(model_type: str, texts: list[str]) -> list[str]:
    """
    Top-level inference function. Loads model and runs prediction.

    Args:
        model_type: 'tfidf' or 'distilbert'
        texts:      list of cleaned review strings

    Returns:
        list of label strings
    """
    model, tokenizer = load_model(model_type)

    if model_type == 'tfidf':
        return predict_tfidf(model, texts)
    else:
        return predict_distilbert(model, tokenizer, texts)