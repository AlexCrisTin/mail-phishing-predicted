import pandas as pd
import re
import warnings
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text

def load_data(csv_path: str = "spam.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Email Text", "Email Type"]).copy()
    df["processed_text"] = df["Email Text"].astype(str).apply(preprocess_text)

    label_mapping = {"Safe Email": 0, "Phishing Email": 1}
    df["label"] = df["Email Type"].map(label_mapping)
    df = df.dropna(subset=["label"]).copy()

    X = df["processed_text"].values
    y = df["label"].astype(int).values
    return X, y

def build_pipeline():
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, min_samples_split=5, min_samples_leaf=2)),
        ]
    )

if __name__ == "__main__":
    X, y = load_data()
    print(f"Successfully loaded {len(X)} data samples!")
    
    model_pipeline = build_pipeline()
    print("Successfully built the model pipeline!")
    print(model_pipeline)