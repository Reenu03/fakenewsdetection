import sys
import types
import pickle
import numpy as np
import streamlit as st

# ── Stub out keras so tokenizer.pkl loads on any Python version ───────────────
class _Tokenizer:
    """Minimal Keras-compatible tokenizer (no keras/tensorflow needed)."""
    def __init__(self):
        pass

    def texts_to_sequences(self, texts):
        results = []
        for text in texts:
            if self.lower:
                text = text.lower()
            words = text.split(self.split)
            seq = []
            for w in words:
                w = w.strip(self.filters)
                if not w:
                    continue
                idx = self.word_index.get(w)
                if idx is None:
                    if self.oov_token:
                        idx = self.word_index.get(self.oov_token, 0)
                    else:
                        continue
                if self.num_words and idx >= self.num_words:
                    continue
                seq.append(idx)
            results.append(seq)
        return results

def _install_keras_stub():
    for mod_name in ['keras', 'keras.preprocessing', 'keras.preprocessing.text']:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules['keras.preprocessing.text'].Tokenizer = _Tokenizer

_install_keras_stub()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article below:")

MAX_LEN = 40

# ── Pure-NumPy helpers ─────────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _relu(x):
    return np.maximum(0, x)

def _pad(seq, maxlen):
    seq = seq[:maxlen]
    return seq + [0] * (maxlen - len(seq))

def _lstm_step(x_t, h, c, W_i, W_h, b):
    z = x_t @ W_i + h @ W_h + b
    u = z.shape[-1] // 4
    i = _sigmoid(z[:u]);  f = _sigmoid(z[u:2*u])
    g = np.tanh(z[2*u:3*u]); o = _sigmoid(z[3*u:])
    c = f * c + i * g
    h = o * np.tanh(c)
    return h, c

def _run_bilstm(seq, ws):
    W_if, W_hf, b_f, W_ib, W_hb, b_b = ws
    units = b_f.shape[-1] // 4
    T = seq.shape[0]
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    for t in range(T):
        h, c = _lstm_step(seq[t], h, c, W_if, W_hf, b_f)
    h_fwd = h
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    for t in reversed(range(T)):
        h, c = _lstm_step(seq[t], h, c, W_ib, W_hb, b_b)
    return np.concatenate([h_fwd, h])

# ── Load resources ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_all():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    w = np.load("model_weights_f16.npz", allow_pickle=True)
    weights = {k: w[k].astype(np.float32) for k in w.files}
    bilstm_w = [weights[k] for k in sorted(k for k in weights if k.startswith("bidirectional"))]

    return (
        tokenizer,
        weights["embedding_0"],
        bilstm_w,
        weights["dense_0"],
        weights["dense_1"],
        weights["dense_1_0"],
        weights["dense_1_1"],
    )

# ── Inference ──────────────────────────────────────────────────────────────────

def predict(text, tokenizer, emb, bilstm_w, dw, db, d1w, d1b):
    seq    = tokenizer.texts_to_sequences([text])[0]
    padded = np.array(_pad(seq, MAX_LEN), dtype=np.int32)
    embedded = emb[padded]
    lstm_out = _run_bilstm(embedded, bilstm_w)
    d1       = _relu(lstm_out @ dw + db)
    logit    = d1 @ d1w + d1b
    return float(_sigmoid(logit[0]))

# ── UI ─────────────────────────────────────────────────────────────────────────

tokenizer, emb, bilstm_w, dw, db, d1w, d1b = load_all()

text = st.text_area("Enter News Text")

if st.button("Predict"):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        score      = predict(text, tokenizer, emb, bilstm_w, dw, db, d1w, d1b)
        confidence = score if score > 0.5 else 1 - score

        if score > 0.5:
            st.error(f"❌ **Fake News** — confidence: {confidence:.0%}")
        else:
            st.success(f"✅ **Real News** — confidence: {confidence:.0%}")

        st.progress(score, text=f"Fake probability: {score:.2%}")
