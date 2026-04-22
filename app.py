import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article below:")

# Load model and tokenizer
@st.cache_resource
def load_all():
    model = load_model("model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    return model, tokenizer

model, tokenizer = load_all()

MAX_LEN = 40

# User input
text = st.text_area("Enter News Text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("⚠ Please enter some text")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        pred = model.predict(padded)[0][0]

        if pred > 0.5:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
