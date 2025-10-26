import pandas as pd
import re
import sys

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) 
    text = text.lower() 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = ' '.join(text.split()) 
    return text

INPUT_FILE = 'train/spam.csv'
OUTPUT_FILE = 'cleaned_spam.csv'

try:
    print(f"Đang tải dữ liệu từ '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'. Dừng chương trình.")
    sys.exit(1)
except Exception as e:
    print(f"Đã xảy ra lỗi khi đọc file: {e}")
    sys.exit(1)

if 'Email Text' not in df.columns or 'Email Type' not in df.columns:
    print(f"Lỗi: File CSV phải chứa cột 'Email Text' và 'Email Type'.")
    sys.exit(1)

df_cleaned = df.dropna(subset=['Email Text', 'Email Type']).copy()

print("Đang làm sạch văn bản...")
df_cleaned['cleaned_text'] = df_cleaned['Email Text'].apply(preprocess_text)

try:
    print(f"Đang lưu dữ liệu đã làm sạch vào '{OUTPUT_FILE}'...")
    df_cleaned_to_save = df_cleaned[['Email Type', 'Email Text', 'cleaned_text']]
    df_cleaned_to_save.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nHoàn tất! Đã tạo file '{OUTPUT_FILE}' thành công.")
except Exception as e:
    print(f"Đã xảy ra lỗi khi lưu file: {e}")