# Unit tests for preprocess and sentiment inference modules.
# Run: pytest tests/test_pipeline.py

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.preprocess import strip_html, remove_special_chars, remove_stopwords, clean_text, preprocess
import pandas as pd


# strip_html

def test_strip_html_removes_tags():
    assert strip_html('<p>Hello world</p>') == ' Hello world '

def test_strip_html_no_tags():
    assert strip_html('plain text') == 'plain text'

def test_strip_html_nested():
    assert strip_html('<div><span>text</span></div>') == '  text  '


# remove_special_chars

def test_remove_special_chars_removes_punctuation():
    result = remove_special_chars('hello, world!')
    assert ',' not in result
    assert '!' not in result

def test_remove_special_chars_collapses_spaces():
    result = remove_special_chars('hello   world')
    assert result == 'hello world'

def test_remove_special_chars_keeps_letters():
    result = remove_special_chars('hello world')
    assert result == 'hello world'


# remove_stopwords

def test_remove_stopwords_removes_common_words():
    result = remove_stopwords('this is a great product')
    assert 'this' not in result.split()
    assert 'is' not in result.split()
    assert 'a' not in result.split()

def test_remove_stopwords_keeps_content_words():
    result = remove_stopwords('this is a great product')
    assert 'great' in result
    assert 'product' in result


# clean_text

def test_clean_text_full_pipeline():
    result = clean_text('<p>This is a GREAT product!</p>')
    assert result == result.lower()
    assert '<' not in result
    assert '!' not in result

def test_clean_text_none_returns_empty():
    assert clean_text(None) == ''

def test_clean_text_empty_string_returns_empty():
    assert clean_text('') == ''

def test_clean_text_whitespace_only_returns_empty():
    assert clean_text('   ') == ''

def test_clean_text_lowercases():
    result = clean_text('HELLO WORLD')
    assert result == result.lower()


# preprocess

def test_preprocess_fills_clean_text_column():
    df = pd.DataFrame({'reviewText': ['Great product!', 'Terrible quality.'], 'label': [2, 0], 'clean_text': ['', '']})
    result = preprocess(df)
    assert result['clean_text'].iloc[0] != ''
    assert result['clean_text'].iloc[1] != ''

def test_preprocess_does_not_mutate_original():
    df = pd.DataFrame({'reviewText': ['Great product!'], 'label': [2], 'clean_text': ['']})
    original_value = df['clean_text'].iloc[0]
    preprocess(df)
    assert df['clean_text'].iloc[0] == original_value

def test_preprocess_handles_null_review_text():
    df = pd.DataFrame({'reviewText': [None, 'Good product'], 'label': [0, 2], 'clean_text': ['', '']})
    result = preprocess(df)
    assert len(result) == 1
    assert result['clean_text'].iloc[0] != ''