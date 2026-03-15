import torch
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.pipeline import Pipeline

from src.config import (
    DISTILBERT_MAX_LEN, DISTILBERT_BATCH_SIZE, LABEL_MAP
)

from constants import (
    TFIDF_VECTORIZER_PATH,
    LR_MODEL_PATH,
    DISTILBERT_DIR
)


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
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        lr_model = joblib.load(LR_MODEL_PATH)
        return Pipeline([('tfidf', vectorizer), ('lr', lr_model)]), None
        
    elif model_type == 'distilbert':
        tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_DIR)
        model = model.to(get_device())
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

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        all_preds.extend(preds)

    return [LABEL_MAP[p] for p in all_preds]
