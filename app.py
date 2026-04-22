import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article below:")

MAX_LEN = 40

# ── Pure-NumPy model ───────────────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def lstm_step(x_t, h, c, W_i, W_h, b):
    """Single LSTM time-step."""
    z = x_t @ W_i + h @ W_h + b          # (4*units,)
    units = z.shape[-1] // 4
    i  = sigmoid(z[..., :units])
    f  = sigmoid(z[..., units:2*units])
    g  = tanh  (z[..., 2*units:3*units])
    o  = sigmoid(z[..., 3*units:])
    c  = f * c + i * g
    h  = o * tanh(c)
    return h, c

def run_bilstm(seq, weights):
    """Bidirectional LSTM: forward + backward, concat final states."""
    W_if, W_hf, b_f = weights[0], weights[1], weights[2]   # forward
    W_ib, W_hb, b_b = weights[3], weights[4], weights[5]   # backward

    units = b_f.shape[-1] // 4
    T = seq.shape[0]

    # Forward pass
    h, c = np.zeros(units, dtype=np.float32), np.zeros(units, dtype=np.float32)
    for t in range(T):
        h, c = lstm_step(seq[t], h, c, W_if, W_hf, b_f)
    h_fwd = h

    # Backward pass
    h, c = np.zeros(units, dtype=np.float32), np.zeros(units, dtype=np.float32)
    for t in reversed(range(T)):
        h, c = lstm_step(seq[t], h, c, W_ib, W_hb, b_b)
    h_bwd = h

    return np.concatenate([h_fwd, h_bwd])   # (256,)

@st.cache_resource
def load_all():
    # Tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Weights (stored as float16, cast to float32 for computation)
    w = np.load("model_weights_f16.npz", allow_pickle=True)
    weights = {k: w[k].astype(np.float32) for k in w.files}

    bilstm_keys = [k for k in w.files if k.startswith("bidirectional")]
    bilstm_w = [weights[k] for k in sorted(bilstm_keys)]   # 0-5

    dense_w  = weights["dense_0"]    # (256, 128)
    dense_b  = weights["dense_1"]    # (128,)
    dense1_w = weights["dense_1_0"]  # (128, 1)
    dense1_b = weights["dense_1_1"]  # (1,)
    emb      = weights["embedding_0"]

    return tokenizer, emb, bilstm_w, dense_w, dense_b, dense1_w, dense1_b

def predict(text, tokenizer, emb, bilstm_w, dense_w, dense_b, dense1_w, dense1_b):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")[0]  # (40,)

    # Embedding lookup
    embedded = emb[padded]   # (40, 128)

    # BiLSTM
    lstm_out = run_bilstm(embedded, bilstm_w)   # (256,)

    # Dense 1
    d1 = relu(lstm_out @ dense_w + dense_b)     # (128,)

    # Dense 2 (output)
    logit = d1 @ dense1_w + dense1_b            # (1,)
    return float(sigmoid(logit[0]))

# ── UI ────────────────────────────────────────────────────────────────────────
tokenizer, emb, bilstm_w, dense_w, dense_b, dense1_w, dense1_b = load_all()

text = st.text_area("Enter News Text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        score = predict(text, tokenizer, emb, bilstm_w, dense_w, dense_b, dense1_w, dense1_b)
        confidence = score if score > 0.5 else 1 - score

        if score > 0.5:
            st.error(f"❌ **Fake News** &nbsp; (confidence: {confidence:.0%})")
        else:
            st.success(f"✅ **Real News** &nbsp; (confidence: {confidence:.0%})")

        st.progress(score, text=f"Fake probability: {score:.2%}")
