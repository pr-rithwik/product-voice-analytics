# product-voice-analytics

> Turn thousands of Amazon reviews into actionable product intelligence — automatically.

---

## The Problem

A product has 8,000 reviews and a 4.2-star rating. You can't read them all. You don't know *why* people are unhappy, or *what* specifically they love. Manual review analysis takes hours and is inconsistent.

## The Solution

Paste a product ASIN or name. Get back:

- **Sentiment breakdown** — % positive, neutral, negative
- **Top praise** — what customers love, in plain English
- **Top complaints** — what's actually driving negative reviews
- **Theme clusters** — specific issues ranked by frequency

Built on real Amazon data (20M+ Electronics reviews). Powered by fine-tuned DistilBERT + topic intelligence layer.

---

## Demo


---

## Architecture


---

## Model Performance


---

## Project Structure

```
product-voice-analytics/
│
├── data/
│   └── raw/                          # place downloaded Amazon files here
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── requirements.txt
└── README.md
```

---

## Roadmap

- [x] Project setup
- [ ] Phase 1: EDA + TF-IDF baseline
- [ ] Phase 2: DistilBERT fine-tuning
- [ ] Phase 3: Topic intelligence layer
- [ ] Phase 4: Streamlit app + HuggingFace deployment
- [ ] Phase 2.5 *(optional)*: GloVe embeddings comparison

---

## Data Source

[Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) — Julian McAuley, UCSD.
Electronics category: 20,994,353 reviews · 786,868 products.

---

## Getting Started

```
git clone https://github.com/pr-rithwik/product-voice-analytics.git
cd product-voice-analytics
pip install -r requirements.txt
```

Place your downloaded Amazon data files in `data/raw/` then start with `notebooks/01_eda.ipynb`.

---

## License

MIT
