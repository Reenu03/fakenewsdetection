import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tf_keras as keras
import numpy as np
import json
import pickle

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model_config.json') as f:
        config = json.load(f)

    model = keras.Sequential.from_config(config)
    data = np.load('model_weights_f16.npz')

    for layer in model.layers:
        weights, i = [], 0
        while f"{layer.name}_{i}" in data:
            weights.append(data[f"{layer.name}_{i}"].astype(np.float32))
            i += 1
        if weights:
            layer.set_weights(weights)

    return model

# ─── Load Tokenizer ───────────────────────────────────────────────────────────
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

MAX_LEN = 40

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("📰 Fake News Detector")
st.write("Enter a news article or headline below to check if it's real or fake.")

text = st.text_area("Enter news text here:", height=150)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        pred = model.predict(padded)[0][0]

        st.markdown("---")
        if pred > 0.5:
            st.error(f"🔴 Fake News ❌  (Confidence: {pred*100:.1f}%)")
        else:
            st.success(f"🟢 Real News ✅  (Confidence: {(1-pred)*100:.1f}%)")
