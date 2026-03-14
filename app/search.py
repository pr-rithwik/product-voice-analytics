# app/search.py — product search and ASIN resolution via DuckDB
import duckdb
import gradio as gr

from src.config import PRODUCT_LOOKUP_PATH


def search_products(query):
    if not query or not isinstance(query, str) or len(query.strip()) < 2:
        return gr.Dropdown(choices=[], value=None)
    result = duckdb.query(f"""
        SELECT title
        FROM '{str(PRODUCT_LOOKUP_PATH)}'
        WHERE title ILIKE '%{query.strip()}%'
        LIMIT 20
    """).df()
    return gr.Dropdown(choices=result['title'].tolist(), value=None)


def resolve_asin(title):
    if not title:
        return ''
    safe_title = title.replace("'", "''")
    result = duckdb.query(f"""
        SELECT asin
        FROM '{str(PRODUCT_LOOKUP_PATH)}'
        WHERE title = '{safe_title}'
        LIMIT 1
    """).df()
    if result.empty:
        return ''
    return result['asin'].iloc[0]