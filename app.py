from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import re

app = Flask(__name__)

model = tf.keras.models.load_model("sentiment_model.keras")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(review_text):
    review_text = clean_text(review_text)
    input_text = np.array([review_text], dtype=object)
    prediction = model.predict(input_text, verbose=0)[0][0]

    print("DEBUG INPUT:", review_text)
    print("DEBUG SCORE:", prediction)

    if prediction >= 0.65:
        return "Positive review", float(prediction)
    return "Negative review", float(prediction)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    review = ""

    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            result, score = predict_sentiment(review)
        else:
            result = "Please enter a review."

    return render_template("index.html", result=result, score=score, review=review)

if __name__ == "__main__":
    app.run(debug=True)