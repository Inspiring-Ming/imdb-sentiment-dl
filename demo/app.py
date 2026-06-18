"""
Streamlit demo for the IMDb sentiment model.

A visitor types a movie review; we reproduce the *exact* training-time
preprocessing (lowercase -> tokenize -> map to Keras imdb word ids, offset by 3
-> pad to maxlen) and run the trained Word2Vec-embedding network to predict
positive vs negative sentiment.

Deployed on Hugging Face Spaces. The trained model (sentiment_pool.keras) ships
alongside this file; the imdb word index is fetched from Keras on first load and
cached.
"""
import re

import numpy as np
import streamlit as st
import tensorflow as tf
from keras.datasets import imdb
from keras.utils import pad_sequences

MAXLEN = 300            # must match config/config.yaml -> data.maxlen
NUM_WORDS = 10000       # must match data.num_words
INDEX_FROM = 3          # keras reserves 0,1,2 for <pad>,<start>,<oov>
MODEL_PATH = "sentiment_pool.keras"

st.set_page_config(page_title="IMDb Sentiment (Deep Learning)", page_icon="🎬")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_word_index():
    # Same mapping the model was trained against.
    return imdb.get_word_index()


def encode(text: str, word_index: dict) -> np.ndarray:
    """Reproduce training preprocessing: text -> padded integer sequence."""
    tokens = re.findall(r"[a-z']+", text.lower())
    ids = []
    for tok in tokens:
        idx = word_index.get(tok)
        # Words outside the top NUM_WORDS (or unknown) -> <oov> token (id 2).
        if idx is not None and idx + INDEX_FROM < NUM_WORDS:
            ids.append(idx + INDEX_FROM)
        else:
            ids.append(2)
    if not ids:
        ids = [2]
    return pad_sequences([ids], maxlen=MAXLEN)


model = load_model()
word_index = load_word_index()

st.title("🎬 IMDb Sentiment Classifier")
st.caption(
    "Word2Vec embeddings → neural network, trained on 25k IMDb reviews "
    "(83.8% test accuracy). Type a movie review and get a live prediction."
)

examples = {
    "— pick an example —": "",
    "Positive": "An absolute masterpiece. The acting was brilliant and the story "
                "kept me on the edge of my seat the whole time.",
    "Negative": "A complete waste of time. The plot made no sense and the acting "
                "was wooden and boring throughout.",
    "Mixed": "The visuals were stunning but the story was painfully slow and the "
             "ending felt rushed and unsatisfying.",
}
choice = st.selectbox("Try an example, or write your own below:", list(examples))
default_text = examples[choice]

review = st.text_area("Movie review", value=default_text, height=160,
                      placeholder="e.g. This film was a thrilling, beautifully acted ride...")

if st.button("Predict sentiment", type="primary") and review.strip():
    x = encode(review, word_index)
    proba = float(model.predict(x, verbose=0).ravel()[0])
    label = "Positive 👍" if proba >= 0.5 else "Negative 👎"
    confidence = proba if proba >= 0.5 else 1 - proba

    st.subheader(label)
    st.progress(proba)
    st.metric("P(positive)", f"{proba:.1%}", help="Model's sigmoid output")
    st.caption(f"Confidence in this call: {confidence:.1%}")

st.divider()
st.markdown(
    "Model & full pipeline: "
    "[github.com/Inspiring-Ming/imdb-sentiment-dl]"
    "(https://github.com/Inspiring-Ming/imdb-sentiment-dl) · "
    "architecture bake-off (MLP vs pooling vs LSTM), leakage-free splits, "
    "full metrics."
)
