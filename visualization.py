"""
File tạo hình ảnh minh họa tổng hợp cho dự án phân tích email spam/phishing
Tạo các biểu đồ trực quan để hiểu rõ dữ liệu và hiệu suất mô hình
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from wordcloud import WordCloud
from collections import Counter
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, roc_curve

# Thiết lập style cho biểu đồ
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản"""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text

def load_data(csv_path: str = "train/spam.csv"):
    """Tải và xử lý dữ liệu"""
    df = pd.read_csv(csv_path)
    if "Email Text" not in df.columns or "Email Type" not in df.columns:
        raise ValueError("Input CSV must contain columns 'Email Text' and 'Email Type'.")

    df = df.dropna(subset=["Email Text", "Email Type"]).copy()
    df["processed_text"] = df["Email Text"].astype(str).apply(preprocess_text)

    label_mapping = {"Safe Email": 0, "Phishing Email": 1}
    df["label"] = df["Email Type"].map(label_mapping)
    df = df.dropna(subset=["label"]).copy()

    X = df["processed_text"].values
    y = df["label"].astype(int).values
    return X, y, df

def create_data_overview_plots(df, output_dir="plots"):
    """Tạo các biểu đồ tổng quan về dữ liệu"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Phân phối loại email
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Phân phối loại email
    plt.subplot(2, 2, 1)
    email_counts = df['Email Type'].value_counts()
    colors = ['#2E8B57', '#DC143C']  # Xanh lá cho Safe, Đỏ cho Phishing
    plt.pie(email_counts.values, labels=email_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Email Type Distribution', fontsize=14, fontweight='bold')
    
    # Subplot 2: Biểu đồ cột phân phối
    plt.subplot(2, 2, 2)
    bars = plt.bar(email_counts.index, email_counts.values, color=colors)
    plt.title('Email Count by Type', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Thêm số liệu lên các cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 3: Độ dài văn bản theo loại
    plt.subplot(2, 2, 3)
    df['text_length'] = df['Email Text'].str.len()
    safe_lengths = df[df['Email Type'] == 'Safe Email']['text_length']
    phishing_lengths = df[df['Email Type'] == 'Phishing Email']['text_length']
    
    plt.hist([safe_lengths, phishing_lengths], bins=30, alpha=0.7, 
             label=['Safe Email', 'Phishing Email'], color=colors)
    plt.title('Email Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Character Length')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Subplot 4: Số từ theo loại
    plt.subplot(2, 2, 4)
    df['word_count'] = df['Email Text'].str.split().str.len()
    safe_words = df[df['Email Type'] == 'Safe Email']['word_count']
    phishing_words = df[df['Email Type'] == 'Phishing Email']['word_count']
    
    plt.hist([safe_words, phishing_words], bins=30, alpha=0.7,
             label=['Safe Email', 'Phishing Email'], color=colors)
    plt.title('Word Count Distribution in Emails', fontsize=14, fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created data overview plot: {output_dir}/data_overview.png")

def create_wordclouds(df, output_dir="plots"):
    """Tạo word cloud cho từng loại email"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Word cloud cho Safe Email
    safe_text = ' '.join(df[df['Email Type'] == 'Safe Email']['processed_text'])
    if safe_text.strip():
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        wordcloud_safe = WordCloud(width=800, height=400, 
                                 background_color='white',
                                 colormap='Greens',
                                 max_words=100).generate(safe_text)
        plt.imshow(wordcloud_safe, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Keywords - Safe Email', fontsize=14, fontweight='bold')
        
        # Word cloud cho Phishing Email
        phishing_text = ' '.join(df[df['Email Type'] == 'Phishing Email']['processed_text'])
        plt.subplot(1, 2, 2)
        wordcloud_phishing = WordCloud(width=800, height=400,
                                     background_color='white',
                                     colormap='Reds',
                                     max_words=100).generate(phishing_text)
        plt.imshow(wordcloud_phishing, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Keywords - Phishing Email', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/wordclouds_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created wordcloud comparison: {output_dir}/wordclouds_comparison.png")

def create_model_comparison_plot(X, y, output_dir="plots"):
    """Tạo biểu đồ so sánh hiệu suất các mô hình"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(alpha=1.0)
    }
    
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
            ('clf', model)
        ])
        
        # Cross validation scores
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
        
        # Train test split for detailed metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
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
        plt.title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        plt.ylabel(metric_name)
        plt.ylim(0, 1)
        
        # Thêm giá trị lên các cột
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created model comparison plot: {output_dir}/model_comparison.png")
    
    return results

def create_feature_importance_plot(X, y, output_dir="plots"):
    """Tạo biểu đồ tầm quan trọng của đặc trưng"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sử dụng Random Forest để lấy feature importance
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    # Lấy feature names và importance
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    importances = pipeline.named_steps['clf'].feature_importances_
    
    # Sắp xếp theo importance
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Most Important Features', fontsize=16, fontweight='bold')
    plt.bar(range(len(indices)), importances[indices], color='#FF6B6B')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created feature importance plot: {output_dir}/feature_importance.png")

def create_confusion_matrices(X, y, output_dir="plots"):
    """Tạo ma trận confusion cho tất cả mô hình"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(alpha=1.0)
    }
    
    plt.figure(figsize=(15, 5))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for i, (name, model) in enumerate(models.items()):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.subplot(1, 3, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Safe', 'Phishing'],
                   yticklabels=['Safe', 'Phishing'])
        plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created confusion matrices: {output_dir}/confusion_matrices.png")

def create_roc_curves(X, y, output_dir="plots"):
    """Tạo đường cong ROC cho tất cả mô hình"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(alpha=1.0)
    }
    
    plt.figure(figsize=(10, 8))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (name, model) in enumerate(models.items()):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created ROC curves: {output_dir}/roc_curves.png")

def generate_all_visualizations():
    """Tạo tất cả các hình ảnh minh họa"""
    print("Bat dau tao cac hinh anh minh hoa...")
    
    # Tải dữ liệu
    try:
        X, y, df = load_data("train/spam.csv")
        print(f"Da tai du lieu: {len(df)} emails")
    except Exception as e:
        print(f"Loi khi tai du lieu: {e}")
        return
    
    # Tạo thư mục plots
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tạo các biểu đồ
    try:
        create_data_overview_plots(df, output_dir)
        create_wordclouds(df, output_dir)
        model_results = create_model_comparison_plot(X, y, output_dir)
        create_feature_importance_plot(X, y, output_dir)
        create_confusion_matrices(X, y, output_dir)
        create_roc_curves(X, y, output_dir)
        
        print(f"\nHoan thanh! Da tao tat ca hinh anh minh hoa trong thu muc '{output_dir}/'")
        print("\nCac file da tao:")
        print("  - data_overview.png: Tong quan ve du lieu")
        print("  - wordclouds_comparison.png: So sanh tu khoa")
        print("  - model_comparison.png: So sanh hieu suat mo hinh")
        print("  - feature_importance.png: Tam quan trong dac trung")
        print("  - confusion_matrices.png: Ma tran confusion")
        print("  - roc_curves.png: Duong cong ROC")
        
        # In kết quả so sánh mô hình
        print(f"\nKet qua so sanh mo hinh:")
        for model_name, metrics in model_results.items():
            print(f"\n{model_name}:")
            print(f"  - Accuracy: {metrics['accuracy']:.3f}")
            print(f"  - Precision: {metrics['precision']:.3f}")
            print(f"  - Recall: {metrics['recall']:.3f}")
            print(f"  - F1-Score: {metrics['f1']:.3f}")
            print(f"  - CV F1-Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
    except Exception as e:
        print(f"Loi khi tao bieu do: {e}")

if __name__ == "__main__":
    generate_all_visualizations()
