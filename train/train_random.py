import re
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

def main():
    logging.info("=== BEGINNING RANDOM FOREST ===")

    df = pd.read_csv("train/spam.csv")
    logging.info(f"Tổng {len(df)} dòng, cột: {list(df.columns)}")
    logging.info(f"Phân bố loại email:\n{df['Email Type'].value_counts()}")

    df = df.dropna(subset=["Email Text", "Email Type"]).copy()
    df["Email Text"] = df["Email Text"].astype(str)
    df["processed_text"] = df["Email Text"].apply(preprocess_text)
    df["text_length"] = df["processed_text"].apply(len)
    df["word_count"] = df["processed_text"].apply(lambda x: len(x.split()))

    label_mapping = {"Safe Email": 0, "Phishing Email": 1}
    df["label"] = df["Email Type"].map(label_mapping)

    logging.info("Phân bố nhãn sau khi chuyển:")
    logging.info(f"\n{df['label'].value_counts()}")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_text = vectorizer.fit_transform(df["processed_text"])
    X_other = df[["text_length", "word_count"]].values
    X = np.hstack([X_text.toarray(), X_other])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logging.info("\n\n\n=== SUMMARY ===")
    logging.info(f"\nAccuracy: {acc:.4f}")
    logging.info(f"\nConfusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"\n{classification_report(y_test, y_pred)}")

    joblib.dump(model, "train/spam_classifier_model.pkl")
    joblib.dump(vectorizer, "train/spam_tfidf_vectorizer.pkl")
    logging.info("Đã lưu mô hình và vectorizer vào thư mục /train.")


if __name__ == "__main__":
    main()
