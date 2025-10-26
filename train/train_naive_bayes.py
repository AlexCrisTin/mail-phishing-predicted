import pandas as pd
import re
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def load_and_prepare_data(file_path: str):
    logging.info("Đang đọc dữ liệu từ CSV...")
    df = pd.read_csv(file_path)

    logging.info(f"Cột dữ liệu: {df.columns.tolist()}")
    logging.info(f"Thống kê Email Type:\n{df['Email Type'].value_counts()}")

    df = df.dropna(subset=['Email Text', 'Email Type'])
    df['Email Text'] = df['Email Text'].astype(str)
    df['processed_text'] = df['Email Text'].apply(preprocess_text)

    label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
    df['label'] = df['Email Type'].map(label_mapping)

    logging.info("Phân bố nhãn sau chuyển đổi:")
    logging.info(f"\n{df['label'].value_counts()}")
    return df

def vectorize_text(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']
    return X, y, vectorizer


def train_model(X_train, y_train):
    logging.info("Training...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    logging.info("\nClassification report:\n" + classification_report(y_test, y_pred))
    logging.info("Confusion matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    logging.info(f"Saved model at: {model_path}")
    logging.info(f"Saved vectorizer at: {vectorizer_path}")

def main():
    data_path = 'train/spam.csv'
    model_path = 'trainspam_naive_bayes_model.pkl'
    vectorizer_path = 'train/spam_naive_bayes_vectorizer.pkl'

    df = load_and_prepare_data(data_path)
    X, y, vectorizer = vectorize_text(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    logging.info("\n\n=== SUMMARY ===")
    evaluate_model(model, X_test, y_test)
    save_model(model, vectorizer, model_path, vectorizer_path)

if __name__ == "__main__":
    main()
