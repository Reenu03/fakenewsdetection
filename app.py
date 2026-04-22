import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

MAX_LEN = 40

st.title("📰 Fake News Detection using LSTM")

input_text = st.text_area("Enter News Content")

if st.button("Predict"):
    if input_text.strip() != "":
        
        input_text = input_text.lower()

        seq = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        st.write(f"Confidence: {prediction*100:.2f}%")

    else:
        st.warning("Please enter some text")
