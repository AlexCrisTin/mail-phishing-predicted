from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def build_pipeline():
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=5000,      
                stop_words="english",  
                ngram_range=(1, 2) 
            )),

            ("clf", RandomForestClassifier(
                n_estimators=200,   
                random_state=42,      
                max_depth=20,          
                n_jobs=-1      
            )),
        ]
    )