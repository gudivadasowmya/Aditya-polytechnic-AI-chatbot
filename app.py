 
import json
import pickle
import random

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open("intents.json", "r", encoding="utf-8") as file:
    intents_data = json.load(file)

with open("chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)

intent_map = {
    intent["tag"]: intent["responses"]
    for intent in intents_data["intents"]
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "reply": "Please enter a question."
        }), 400

    tag = model.predict([message])[0]
    confidence = max(model.predict_proba([message])[0])

    if confidence < 0.25:
        reply = (
            "Sorry, I don't have a reliable answer for that question. "
            "Please ask about courses, admission, fees or college timings."
        )
    else:
        reply = random.choice(intent_map[tag])

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)

