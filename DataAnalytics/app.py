from flask import Flask, request, jsonify
import joblib

# Загружаем модель и TF-IDF
model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Flask API
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' in request."}), 400

    text = data["text"]
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    label = "POSITIVE" if prediction == 1 else "NEGATIVE"

    return jsonify({"text": text, "prediction": label})

if __name__ == "__main__":
    app.run(debug=True)