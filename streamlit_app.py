import os
import json
import textwrap
import time

import httpx
import streamlit as st
from transformers import pipeline

# --- CONFIG -----------------------------------------------------------------

MODEL_ID = os.getenv("MODEL_ID", "sshleifer/distilbart-cnn-12-6")

# HF token: on Streamlit Cloud, put this in st.secrets["HF_TOKEN"].
# Locally, just use the HF_TOKEN environment variable.
HF_TOKEN = os.getenv("HF_TOKEN", None)

try:
    # This will only work if secrets.toml exists (e.g., on Streamlit Cloud)
    HF_TOKEN = st.secrets["HF_TOKEN"]  # type: ignore[attr-defined]
except Exception:
    # No secrets configured -> rely on env var
    pass

# --- SUMMARIZATION BACKENDS -------------------------------------------------

@st.cache_resource
def get_local_summarizer(model_id: str):
    """
    Cached local summarization pipeline so the model loads only once.
    """
    return pipeline("summarization", model=model_id)


def summarize_local(text: str, min_length: int, max_length: int) -> tuple[str, int]:
    summarizer = get_local_summarizer(MODEL_ID)
    t0 = time.perf_counter()
    out = summarizer(
        text,
        min_length=min_length,
        max_length=max_length,
        truncation=True,
    )[0]["summary_text"]
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return out, latency_ms


def summarize_hf(text: str, min_length: int, max_length: int) -> tuple[str, int]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Please configure it in env or st.secrets.")

    url = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {
            "min_length": min_length,
            "max_length": max_length,
        },
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # HF can return list or list-of-lists
    if data and isinstance(data[0], list):
        data = data[0]
    summary_text = data[0]["summary_text"]
    return summary_text, latency_ms


# --- STREAMLIT UI -----------------------------------------------------------

st.set_page_config(page_title="Summarization Service Demo", layout="centered")

st.title("🧠 Summarization Demo")
st.caption("Single-file Streamlit app using a pretrained summarization model")

st.markdown("### Backend")
backend = st.radio(
    "Choose where to run summarization",
    options=["hf", "local"],
    format_func=lambda x: "Hugging Face Inference API (cloud)" if x == "hf" else "Local model (inside app)",
    horizontal=False,
)

st.markdown("### Input text")

sample_text = textwrap.dedent(
    """
    Paste or type a paragraph here. For example:

    OpenAI recently released a new model that improves inference speed and reduces cost, 
    allowing developers to build AI applications that respond faster and scale more 
    efficiently. Companies are experimenting with integrating these models into their 
    existing products to enhance user experiences and unlock new capabilities.
    """
).strip()

text = st.text_area(
    "Text to summarize",
    value=sample_text,
    height=220,
)

st.markdown("### Parameters")
col1, col2 = st.columns(2)
with col1:
    min_length = st.slider("Minimum summary length", 10, 200, 30, 5)
with col2:
    max_length = st.slider("Maximum summary length", 40, 512, 120, 10)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        payload = {
            "backend": backend,
            "text": text,
            "min_length": min_length,
            "max_length": max_length,
        }

        st.markdown("#### Request payload (for reference)")
        st.code(json.dumps(payload, indent=2), language="json")

        try:
            with st.spinner(f"Running summarization via {backend.upper()}..."):
                if backend == "hf":
                    summary, latency_ms = summarize_hf(
                        text=text,
                        min_length=min_length,
                        max_length=max_length,
                    )
                else:
                    summary, latency_ms = summarize_local(
                        text=text,
                        min_length=min_length,
                        max_length=max_length,
                    )
        except Exception as e:
            st.error(f"Error during summarization: {e}")
        else:
            st.markdown("### ✅ Summary")
            st.write(summary)

            st.markdown("### ⚙️ Metadata")
            meta = {
                "backend": backend,
                "model": MODEL_ID,
                "latency_ms": latency_ms,
            }
            st.json(meta)
