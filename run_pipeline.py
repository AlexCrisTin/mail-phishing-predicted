import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pipeline import full_spam_pipeline

print("Bắt đầu quy trình: Huấn luyện -> Dự đoán -> Xuất file...")
print("Đang tải và chuẩn bị dữ liệu...")
df = pd.read_csv('spam.csv')
df = df.dropna(subset=['Email Text', 'Email Type'])
df['processed_text'] = df['Email Text'].astype(str)
df['text_length'] = df['Email Text'].astype(str).apply(len)
df['word_count'] = df['Email Text'].astype(str).apply(lambda x: len(x.split()))
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
df['label'] = df['Email Type'].map(label_mapping)
df = df.dropna(subset=['label'])
X = df[['processed_text', 'text_length', 'word_count']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Đang huấn luyện pipeline trên tập dữ liệu train...")
pipeline_da_huan_luyen = full_spam_pipeline
pipeline_da_huan_luyen.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")
joblib.dump(pipeline_da_huan_luyen, 'trained_spam_pipeline.pkl')
print("Đã lưu pipeline đã huấn luyện vào file 'trained_spam_pipeline.pkl'")
print("Đang thực hiện dự đoán trên toàn bộ dữ liệu...")
predictions = pipeline_da_huan_luyen.predict(X)
print("Đang xuất kết quả ra file CSV...")
reverse_mapping = {0: 'Safe Email', 1: 'Phishing Email'}
df['predicted_type'] = [reverse_mapping[p] for p in predictions]
output_df = df[['Email Text', 'Email Type', 'predicted_type']]
output_df.to_csv('spam_pipeline.csv', index=False)
print("\nHoàn tất!")
print("Đã xuất kết quả dự đoán ra file 'spam_pipeline.csv'")