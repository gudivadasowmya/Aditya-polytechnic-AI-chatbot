 
import json
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent

# Load training questions
with open(BASE_DIR / "intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

questions = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        questions.append(pattern.lower().strip())
        labels.append(intent["tag"])

# Combine word-based and character-based features
features = FeatureUnion([
    (
        "word_features",
        TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "character_features",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        )
    )
])

# AI classification model
model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

# Train the model
model.fit(questions, labels)

# Save the trained model
joblib.dump(model, BASE_DIR / "chatbot_model.pkl")

print("AI model trained successfully!")
print("Total training questions:", len(questions))
print("Total intent categories:", len(set(labels)))
print("Model saved as chatbot_model.pkl")
