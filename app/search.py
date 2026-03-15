# app/search.py — product search and ASIN resolution via DuckDB
import duckdb
import gradio as gr

from constants import PRODUCT_LOOKUP_PATH


def search_products(query):
    if not query or not isinstance(query, str) or len(query.strip()) < 2:
        return gr.Dropdown(choices=[], value=None)
    q = query.strip()
    result = duckdb.query(f"""
        SELECT title
        FROM '{str(PRODUCT_LOOKUP_PATH)}'
        WHERE title ILIKE '%{q}%'
           OR asin ILIKE '%{q}%'
        ORDER BY review_count DESC
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