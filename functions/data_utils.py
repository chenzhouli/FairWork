# data_utils.py
import json
import gzip
import ast
from pathlib import Path
import streamlit as st
from config import DATA_DIR

def restore_keys_recursively(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            try:
                new_key = ast.literal_eval(k)
            except (ValueError, SyntaxError):
                new_key = k
            new_dict[new_key] = restore_keys_recursively(v)
        return new_dict
    elif isinstance(obj, list):
        return [restore_keys_recursively(i) for i in obj]
    else:
        return obj

@st.cache_data
def load_overview_metrics() -> dict:
    with open(DATA_DIR / "overview_metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_overview_bias() -> dict:
    with open(DATA_DIR / "overview_bias.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_details(model_key: str) -> dict:
    with gzip.open(DATA_DIR / model_key / "details.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)
    