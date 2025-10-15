import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from scipy.sparse import hstack

# 2. Đọc dữ liệu
print("Loading data...")
df = pd.read_csv('steam_reviews_custom.csv')

print("Dữ liệu mẫu:")
print(df.head())
print("\nCác cột trong dataset:", df.columns)
print("\nThống kê rating:", df['rating'].value_counts())

# 3. Loại bỏ missing data
df = df.dropna(subset=['review', 'rating'])
df['review'] = df['review'].astype(str)

# 4. Xử lý số giờ chơi
if 'hour_played' in df.columns:
    df['playtime'] = pd.to_numeric(df['hour_played'], errors='coerce').fillna(0)
    print("Đã sử dụng cột số giờ chơi: hour_played")
else:
    df['playtime'] = 0
    print("Không tìm thấy cột số giờ chơi, sẽ dùng mặc định 0.")

# 5. Feature engineering
df['review_length'] = df['review'].apply(len)
df['word_count'] = df['review'].apply(lambda x: len(x.split()))

# 6. Text vectorization
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_text = vectorizer.fit_transform(df['review'])

# 7. Kết hợp các đặc trưng (bao gồm playtime)
X_other = df[['review_length', 'word_count', 'playtime']].values
X = hstack([X_text, X_other])
y = df['rating']

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Huấn luyện mô hình
print("Training model...")
model = RandomForestClassifier(n_estimators=20,random_state=42)
model.fit(X_train, y_train)

# 10. Đánh giá mô hình
print("\nEvaluating model...")
y_pred = model.predict(X_test)
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 11. Lưu model và vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(model, 'game_rating_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# 12. Xuất file kết quả dự đoán mẫu
df_sample = df.sample(10, random_state=1)
sample_X_text = vectorizer.transform(df_sample['review'])
sample_X_other = df_sample[['review_length', 'word_count', 'playtime']].values
sample_X = hstack([sample_X_text, sample_X_other])
sample_pred = model.predict(sample_X)
df_sample['predicted_rating'] = sample_pred
df_sample.to_csv('sample_pred_results.csv', index=False)
print("\nĐã xuất file sample_pred_results.csv với dự đoán mẫu.")

# 13. Hàm dự đoán rating cho review mới 
def predict_review_rating(review_text, playtime=0):
    """Dự đoán rating cho 1 review mới (chuỗi), kèm số giờ chơi."""
    model = joblib.load('game_rating_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    review_length = len(review_text)
    word_count = len(review_text.split())
    X_text = vectorizer.transform([review_text])
    X_other = np.array([[review_length, word_count, playtime]])
    X = hstack([X_text, X_other])
    pred = model.predict(X)
    return int(pred[0])

# Test thử:
test_review = "Nice"
test_playtime = 189  # Giả sử chơi 189 tiếng
print(f"\nTest với review: {test_review} | Số giờ chơi: {test_playtime}")
print("Predicted rating:", predict_review_rating(test_review, playtime=test_playtime))

print("Done!") 