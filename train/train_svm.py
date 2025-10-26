import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import re
import string

#read data
df = pd.read_csv('spam.csv')
print(df.head())
print("\nDataset columns:", df.columns)
print("\nEmail type statistics:", df['Email Type'].value_counts())

df = df.dropna(subset=['Email Text', 'Email Type'])
df['Email Text'] = df['Email Text'].astype(str)

# clean text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# processed text
df['processed_text'] = df['Email Text'].apply(preprocess_text)

# text length
df['text_length'] = df['processed_text'].apply(len)
df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))

# label mapping
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
df['label'] = df['Email Type'].map(label_mapping)

# label distribution
print("\nLabel distribution after conversion:")
print(df['label'].value_counts())

#tfidf
vectorizer = TfidfVectorizer(
    max_features=5000, 
    stop_words='english',
    ngram_range=(1, 2),  
    min_df=2,  
    max_df=0.95  
)
X_text = vectorizer.fit_transform(df['processed_text'])

X_other = df[['text_length', 'word_count']].values
X = np.hstack([X_text.toarray(), X_other])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SVM train
model = SVC(
    random_state=42,
    kernel='rbf',
    C=1.0,
    gamma='scale'
)
model.fit(X_train, y_train)

# accuracy
y_pred = model.predict(X_test)
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# save model
joblib.dump(model, 'spam_svm_model.pkl')
joblib.dump(vectorizer, 'spam_svm_vectorizer.pkl')
