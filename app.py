from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# load model and tokenizer
model = load_model("model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

MAX_LEN = 40

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        text = request.form["news"]

        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        pred = model.predict(padded)[0][0]

        if pred > 0.5:
            prediction = "Fake News ❌"
        else:
            prediction = "Real News ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)