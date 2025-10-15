import argparse
import joblib
import numpy as np
import pandas as pd
import re


def preprocess_text(text: str) -> str:
    """Lowercase, remove non-letters, normalize spaces."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text


def load_artifacts(model_path: str = "spam_classifier_model.pkl", vectorizer_path: str = "spam_tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_text(model, vectorizer, email_text: str) -> dict:
    processed = preprocess_text(email_text)
    text_length = len(processed)
    word_count = len(processed.split())
    X_text = vectorizer.transform([processed])
    X_other = np.array([[text_length, word_count]])
    X = np.hstack([X_text.toarray(), X_other])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = "Phishing Email" if pred == 1 else "Safe Email"
    return {
        "prediction": label,
        "phishing_probability": float(proba[1]),
        "safe_probability": float(proba[0]),
    }


def predict_csv(model, vectorizer, input_csv: str, text_col: str = "Email Text", out_csv: str = "predictions.csv") -> str:
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {input_csv}")
    df_proc = df.copy()
    df_proc["processed_text"] = df_proc[text_col].astype(str).apply(preprocess_text)
    df_proc["text_length"] = df_proc["processed_text"].apply(len)
    df_proc["word_count"] = df_proc["processed_text"].apply(lambda x: len(x.split()))
    X_text = vectorizer.transform(df_proc["processed_text"])
    X_other = df_proc[["text_length", "word_count"]].values
    X = np.hstack([X_text.toarray(), X_other])
    preds = model.predict(X)
    probas = model.predict_proba(X)
    reverse_mapping = {0: "Safe Email", 1: "Phishing Email"}
    df_out = df.copy()
    df_out["predicted_type"] = [reverse_mapping[p] for p in preds]
    df_out["phishing_probability"] = probas[:, 1]
    df_out["safe_probability"] = probas[:, 0]
    df_out.to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Test spam/phishing email classifier")
    parser.add_argument("-t", "--text", type=str, help="Email text to classify")
    parser.add_argument("-f", "--file", type=str, help="CSV file to classify (expects column 'Email Text')")
    parser.add_argument("--text-col", type=str, default="Email Text", help="Text column name in CSV")
    parser.add_argument("-o", "--output", type=str, default="predictions.csv", help="Output CSV path for batch mode")
    parser.add_argument("--model", type=str, default="spam_classifier_model.pkl", help="Path to model .pkl")
    parser.add_argument("--vectorizer", type=str, default="spam_tfidf_vectorizer.pkl", help="Path to vectorizer .pkl")
    args = parser.parse_args()

    model, vectorizer = load_artifacts(args.model, args.vectorizer)

    if args.text:
        result = predict_text(model, vectorizer, args.text)
        print("Prediction:", result["prediction"])
        print("Phishing Probability:", f"{result['phishing_probability']:.3f}")
        print("Safe Probability:", f"{result['safe_probability']:.3f}")
        return

    if args.file:
        out_path = predict_csv(model, vectorizer, args.file, text_col=args.text_col, out_csv=args.output)
        print(f"Saved predictions to {out_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()


