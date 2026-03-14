# app/ui.py — gradio UI components and event bindings
import gradio as gr
from functools import partial

from app import search_products, analyse


def build_ui(tfidf_pipeline, distilbert_model, distilbert_tokenizer, demo_cache, product_names):
    dropdown_choices = ['-- Select a demo product --'] + list(product_names.keys()) + ['Custom Search']

    analyse_fn = partial(
        analyse,
        tfidf_pipeline=tfidf_pipeline,
        distilbert_model=distilbert_model,
        distilbert_tokenizer=distilbert_tokenizer,
        demo_cache=demo_cache,
        product_names=product_names
    )

    with gr.Blocks(title='Product Voice Analytics') as demo:
        gr.Markdown('# 🔍 Product Voice Analytics')
        gr.Markdown('Select a demo product for instant results, or search any Amazon Electronics product for live analysis.')

        with gr.Row():
            product_dropdown = gr.Dropdown(
                choices=dropdown_choices,
                label='Demo Products',
                value=dropdown_choices[1]
            )
            model_choice = gr.Radio(
                ['TF-IDF + LR', 'DistilBERT'],
                label='Sentiment Model',
                value='TF-IDF + LR'
            )

        with gr.Accordion('Advanced — Analyse any product (may take several minutes)', open=False):
            product_search = gr.Textbox(
                label='Search Product by Name',
                placeholder='Type to search e.g. Sony headphones'
            )
            search_results = gr.Dropdown(
                choices=[],
                label='Select from results',
                interactive=True
            )
            gr.Markdown('⚠️ Only products with 50+ reviews are shown. Streams the full dataset. Best used with TF-IDF + LR for speed.')

        analyse_btn   = gr.Button('Analyse', variant='primary')
        status_out    = gr.Textbox(label='Status',              interactive=False)
        sentiment_out = gr.Textbox(label='Sentiment Breakdown', interactive=False, lines=6)

        with gr.Row():
            praise_out    = gr.Textbox(label='✅ Top Praise Themes',    interactive=False, lines=8)
            complaint_out = gr.Textbox(label='⚠️ Top Complaint Themes', interactive=False, lines=8)

        product_search.change(
            fn=search_products,
            inputs=[product_search],
            outputs=[search_results]
        )

        analyse_btn.click(
            fn=analyse_fn,
            inputs=[product_dropdown, search_results, model_choice],
            outputs=[status_out, sentiment_out, praise_out, complaint_out]
        )

    return demo