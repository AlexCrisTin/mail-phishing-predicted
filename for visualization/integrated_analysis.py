
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import re
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.model_selection import train_test_split

# Thiết lập style cho biểu đồ
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản giống như trong các file training"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

def load_data_and_prepare():
    """Load và chuẩn bị dữ liệu giống như trong các file training"""
    df = pd.read_csv("train/spam.csv")
    df = df.dropna(subset=["Email Text", "Email Type"]).copy()
    df["Email Text"] = df["Email Text"].astype(str)
    df["processed_text"] = df["Email Text"].apply(preprocess_text)
    
    # Thêm các đặc trưng như trong file training
    df["text_length"] = df["processed_text"].apply(len)
    df["word_count"] = df["processed_text"].apply(lambda x: len(x.split()))
    
    label_mapping = {"Safe Email": 0, "Phishing Email": 1}
    df["label"] = df["Email Type"].map(label_mapping)
    
    return df

def load_models():
    """Load tất cả các mô hình đã được train"""
    models = {}
    
    try:
        # Load Random Forest model
        rf_model = joblib.load("train/spam_classifier_model.pkl")
        rf_vectorizer = joblib.load("train/spam_tfidf_vectorizer.pkl")
        models['Random Forest'] = {
            'model': rf_model,
            'vectorizer': rf_vectorizer,
            'type': 'random_forest'
        }
        print("Loaded Random Forest model successfully")
    except Exception as e:
        print(f"Error loading Random Forest model: {e}")
    
    try:
        # Load Logistic Regression model
        lr_model = joblib.load("train/spam_logistic_regression_model.pkl")
        lr_vectorizer = joblib.load("train/spam_logistic_regression_vectorizer.pkl")
        models['Logistic Regression'] = {
            'model': lr_model,
            'vectorizer': lr_vectorizer,
            'type': 'logistic_regression'
        }
        print("Loaded Logistic Regression model successfully")
    except Exception as e:
        print(f"Error loading Logistic Regression model: {e}")
    
    try:
        # Load Naive Bayes model
        nb_model = joblib.load("trainspam_naive_bayes_model.pkl")
        nb_vectorizer = joblib.load("train/spam_naive_bayes_vectorizer.pkl")
        models['Naive Bayes'] = {
            'model': nb_model,
            'vectorizer': nb_vectorizer,
            'type': 'naive_bayes'
        }
        print("Loaded Naive Bayes model successfully")
    except Exception as e:
        print(f"Error loading Naive Bayes model: {e}")
    
    return models

def prepare_features(df, vectorizer, model_type):
    """Chuẩn bị đặc trưng cho từng loại mô hình"""
    if model_type == 'naive_bayes':
        # Naive Bayes chỉ sử dụng text features
        X_text = vectorizer.transform(df["processed_text"])
        return X_text
    else:
        # Random Forest và Logistic Regression sử dụng cả text và numerical features
        X_text = vectorizer.transform(df["processed_text"])
        X_other = df[["text_length", "word_count"]].values
        X = np.hstack([X_text.toarray(), X_other])
        return X

def create_comprehensive_confusion_matrices(models, df, output_dir="photo result"):
    """Tạo confusion matrix cho tất cả mô hình"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Chuẩn bị dữ liệu test
    X_test_data = {}
    y_test = None
    
    for name, model_info in models.items():
        vectorizer = model_info['vectorizer']
        model_type = model_info['type']
        
        # Chuẩn bị features
        X = prepare_features(df, vectorizer, model_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"].values
        )
        
        X_test_data[name] = X_test
    
    # Tạo biểu đồ confusion matrix
    plt.figure(figsize=(15, 5))
    
    for i, (name, model_info) in enumerate(models.items()):
        model = model_info['model']
        X_test = X_test_data[name]
        
        # Predict
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.subplot(1, len(models), i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Safe', 'Phishing'],
                   yticklabels=['Safe', 'Phishing'])
        plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/integrated_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created integrated confusion matrices: {output_dir}/integrated_confusion_matrices.png")

def create_comprehensive_roc_curves(models, df, output_dir="photo result"):
    """Tạo ROC curves cho tất cả mô hình"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (name, model_info) in enumerate(models.items()):
        model = model_info['model']
        vectorizer = model_info['vectorizer']
        model_type = model_info['type']
        
        # Chuẩn bị features
        X = prepare_features(df, vectorizer, model_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"].values
        )
        
        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - Trained Models', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/integrated_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created integrated ROC curves: {output_dir}/integrated_roc_curves.png")

def create_comprehensive_performance_comparison(models, df, output_dir="photo result"):    
    """Tạo biểu đồ so sánh hiệu suất tổng hợp"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    
    for name, model_info in models.items():
        model = model_info['model']
        vectorizer = model_info['vectorizer']
        model_type = model_info['type']
        
        # Chuẩn bị features
        X = prepare_features(df, vectorizer, model_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"].values
        )
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    
    # Tạo biểu đồ so sánh
    plt.figure(figsize=(15, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        model_names = list(results.keys())
        values = [results[name][metric] for name in model_names]
        
        bars = plt.bar(model_names, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title(f'{metric_name} Comparison - Trained Models', fontsize=14, fontweight='bold')
        plt.ylabel(metric_name)
        plt.ylim(0, 1)
        
        # Thêm giá trị lên các cột
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/integrated_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created integrated performance comparison: {output_dir}/integrated_performance_comparison.png")
    
    return results

def create_detailed_classification_reports(models, df, output_dir="photo result"): 
    """Tạo báo cáo phân loại chi tiết cho từng mô hình"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*80)
    
    for name, model_info in models.items():
        model = model_info['model']
        vectorizer = model_info['vectorizer']
        model_type = model_info['type']
        
        # Chuẩn bị features
        X = prepare_features(df, vectorizer, model_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"].values
        )
        
        # Predict
        y_pred = model.predict(X_test)
        
        print(f"\n{name.upper()} MODEL:")
        print("-" * 50)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe Email', 'Phishing Email']))

def generate_integrated_analysis():
    """Tạo tất cả phân tích tích hợp từ các mô hình đã train"""
    print("Starting integrated analysis of trained models...")
    
    # Load data
    df = load_data_and_prepare()
    print(f"Loaded data: {len(df)} emails")
    
    # Load models
    models = load_models()
    if not models:
        print("No models loaded successfully!")
        return
    
    print(f"Loaded {len(models)} models successfully")
    
    # Tạo thư mục plots
    output_dir = "photo result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tạo các biểu đồ tích hợp
    try:
        create_comprehensive_confusion_matrices(models, df, output_dir)
        create_comprehensive_roc_curves(models, df, output_dir)
        results = create_comprehensive_performance_comparison(models, df, output_dir)
        create_detailed_classification_reports(models, df, output_dir)
        
        print(f"\nCompleted integrated analysis!")
        print(f"Generated files in '{output_dir}/':")
        print("  - integrated_confusion_matrices.png")
        print("  - integrated_roc_curves.png")
        print("  - integrated_performance_comparison.png")
        
        # In kết quả tổng hợp
        print(f"\nPERFORMANCE SUMMARY:")
        print("-" * 50)
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  - Accuracy: {metrics['accuracy']:.3f}")
            print(f"  - Precision: {metrics['precision']:.3f}")
            print(f"  - Recall: {metrics['recall']:.3f}")
            print(f"  - F1-Score: {metrics['f1']:.3f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    generate_integrated_analysis()
