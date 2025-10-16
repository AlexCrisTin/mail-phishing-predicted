import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import re
import string

# 1. Đọc dữ liệu
print("Loading data...")
df = pd.read_csv('spam.csv')

print("Sample data:")
print(df.head())
print("\nDataset columns:", df.columns)
print("\nEmail type statistics:", df['Email Type'].value_counts())

# 2. Loại bỏ missing data và chuẩn hóa
df = df.dropna(subset=['Email Text', 'Email Type'])
df['Email Text'] = df['Email Text'].astype(str)

# 3. Tiền xử lý text
def preprocess_text(text):
    """Tiền xử lý text: chuyển về chữ thường, loại bỏ ký tự đặc biệt"""
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ ký tự đặc biệt và số
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    return text

print("Preprocessing text...")
df['processed_text'] = df['Email Text'].apply(preprocess_text)

# 4. Feature engineering
df['text_length'] = df['processed_text'].apply(len)
df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))

# 5. Chuyển đổi nhãn thành số
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
df['label'] = df['Email Type'].map(label_mapping)

print("\nLabel distribution after conversion:")
print(df['label'].value_counts())

# 6. Text vectorization
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=5000, 
    stop_words='english',
    ngram_range=(1, 2),  # Sử dụng unigram và bigram
    min_df=2,  # Từ phải xuất hiện ít nhất 2 lần
    max_df=0.95  # Từ không được xuất hiện quá 95% documents
)
X_text = vectorizer.fit_transform(df['processed_text'])

# 7. Kết hợp các đặc trưng
X_other = df[['text_length', 'word_count']].values
X = np.hstack([X_text.toarray(), X_other])
y = df['label']

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData size:")
print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# 9. Huấn luyện mô hình
print("Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)
model.fit(X_train, y_train)

# 10. Đánh giá mô hình
print("\nEvaluating model...")
y_pred = model.predict(X_test)
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 11. Lưu model và vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'spam_tfidf_vectorizer.pkl')

# 12. Xuất file kết quả dự đoán mẫu
df_sample = df.sample(20, random_state=1)
sample_X_text = vectorizer.transform(df_sample['processed_text'])
sample_X_other = df_sample[['text_length', 'word_count']].values
sample_X = np.hstack([sample_X_text.toarray(), sample_X_other])
sample_pred = model.predict(sample_X)

# Chuyển đổi ngược nhãn để hiển thị
reverse_mapping = {0: 'Safe Email', 1: 'Phishing Email'}
df_sample['predicted_type'] = [reverse_mapping[pred] for pred in sample_pred]
df_sample[['Email Text', 'Email Type', 'predicted_type']].to_csv('spam_prediction_results.csv', index=False)
print("\nExported spam_prediction_results.csv with sample predictions.")

# 13. Hàm dự đoán email spam
def predict_email_spam(email_text):
    """Dự đoán email có phải spam hay không"""
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('spam_tfidf_vectorizer.pkl')
    
    # Tiền xử lý text
    processed_text = preprocess_text(email_text)
    text_length = len(processed_text)
    word_count = len(processed_text.split())
    
    # Vectorization
    X_text = vectorizer.transform([processed_text])
    X_other = np.array([[text_length, word_count]])
    X = np.hstack([X_text.toarray(), X_other])
    
    # Dự đoán
    pred = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return {
        'prediction': 'Phishing Email' if pred == 1 else 'Safe Email',
        'phishing_probability': probability[1],
        'safe_probability': probability[0]
    }

for i, email in enumerate(test_emails, 1):
    result = predict_email_spam(email)
    print(f"\nEmail {i}: {email[:50]}...")
    print(f"Prediction: {result['prediction']}")
    print(f"Phishing Probability: {result['phishing_probability']:.3f}")
    print(f"Safe Probability: {result['safe_probability']:.3f}")

print("\nDone!") 