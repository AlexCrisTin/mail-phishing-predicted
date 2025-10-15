import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

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

if __name__ == "__main__":
    X, y = load_data()
    print(f"Successfully loaded {len(X)} data samples!")
    print("Example of a processed text sample:", X[0])
    print("Corresponding label:", y[0])