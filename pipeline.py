from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(
    transformers=[
        ('text_vectorizer', 
         TfidfVectorizer(
             max_features=5000, 
             stop_words='english',
             ngram_range=(1, 2),
             min_df=2,
             max_df=0.95
         ), 
         'processed_text' 
        ),

        ('numeric_features', 
         'passthrough', 
         ['text_length', 'word_count'] 
        )
    ],
    remainder='drop' 
)

full_spam_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ]
)

if __name__ == '__main__':
    print("="*50)
    print("Đã tạo thành công đối tượng pipeline.")
    print("Đây là cấu trúc của pipeline:")
    print("="*50)
    print(full_spam_pipeline)