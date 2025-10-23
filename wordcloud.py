import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for word cloud generation"""
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_data(csv_path: str = "train/spam.csv"):
    """Load and preprocess the spam dataset"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check required columns
        if "Email Text" not in df.columns or "Email Type" not in df.columns:
            raise ValueError("Input CSV must contain columns 'Email Text' and 'Email Type'.")
        
        # Clean data
        df = df.dropna(subset=["Email Text", "Email Type"]).copy()
        df["processed_text"] = df["Email Text"].astype(str).apply(preprocess_text)
        
        # Remove empty texts
        df = df[df["processed_text"].str.len() > 0].copy()
        
        print(f"After cleaning: {len(df)} rows")
        print(f"Email types: {df['Email Type'].value_counts().to_dict()}")
        
        return df
    except FileNotFoundError:
        print(f"File {csv_path} not found. Please check the file path.")
        return None

def create_word_cloud(text_data, title="Word Cloud", max_words=100, width=800, height=400):
    """Create a word cloud from text data"""
    # Combine all text
    if isinstance(text_data, pd.Series):
        combined_text = ' '.join(text_data.tolist())
    else:
        combined_text = ' '.join(text_data)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=max_words,
        colormap='viridis',
        relative_scaling=0.5,
        random_state=42
    ).generate(combined_text)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return wordcloud

def get_top_words(text_data, n=20):
    """Get top N most frequent words"""
    if isinstance(text_data, pd.Series):
        combined_text = ' '.join(text_data.tolist())
    else:
        combined_text = ' '.join(text_data)
    
    # Split into words and count
    words = combined_text.split()
    word_counts = Counter(words)
    
    # Remove very short words
    word_counts = Counter({word: count for word, count in word_counts.items() if len(word) > 2})
    
    return word_counts.most_common(n)

def plot_word_frequency(word_counts, title="Top Words", n=20):
    """Plot word frequency bar chart"""
    top_words = dict(word_counts[:n])
    
    plt.figure(figsize=(12, 8))
    words = list(top_words.keys())
    counts = list(top_words.values())
    
    bars = plt.bar(range(len(words)), counts, color='skyblue', alpha=0.7)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()

def generate_comparative_wordclouds(df, output_path="wordcloud_analysis.png"):
    """Generate comparative word clouds for spam vs safe emails"""
    # Separate spam and safe emails
    spam_emails = df[df['Email Type'] == 'Phishing Email']['processed_text']
    safe_emails = df[df['Email Type'] == 'Safe Email']['processed_text']
    
    print(f"Spam emails: {len(spam_emails)}")
    print(f"Safe emails: {len(safe_emails)}")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Email Text Analysis - Word Clouds and Frequency', fontsize=16, fontweight='bold')
    
    # Spam word cloud
    ax1 = axes[0, 0]
    spam_wordcloud = WordCloud(
        width=400, height=300,
        background_color='white',
        max_words=100,
        colormap='Reds',
        relative_scaling=0.5,
        random_state=42
    ).generate(' '.join(spam_emails.tolist()))
    
    ax1.imshow(spam_wordcloud, interpolation='bilinear')
    ax1.set_title('Phishing/Spam Emails Word Cloud', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Safe word cloud
    ax2 = axes[0, 1]
    safe_wordcloud = WordCloud(
        width=400, height=300,
        background_color='white',
        max_words=100,
        colormap='Greens',
        relative_scaling=0.5,
        random_state=42
    ).generate(' '.join(safe_emails.tolist()))
    
    ax2.imshow(safe_wordcloud, interpolation='bilinear')
    ax2.set_title('Safe Emails Word Cloud', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Spam word frequency
    ax3 = axes[1, 0]
    spam_words = get_top_words(spam_emails, n=15)
    spam_word_counts = dict(spam_words)
    
    words = list(spam_word_counts.keys())
    counts = list(spam_word_counts.values())
    bars = ax3.bar(range(len(words)), counts, color='red', alpha=0.7)
    ax3.set_title('Top Words in Phishing/Spam Emails', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Words')
    ax3.set_ylabel('Frequency')
    ax3.set_xticks(range(len(words)))
    ax3.set_xticklabels(words, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontsize=8)
    
    # Safe word frequency
    ax4 = axes[1, 1]
    safe_words = get_top_words(safe_emails, n=15)
    safe_word_counts = dict(safe_words)
    
    words = list(safe_word_counts.keys())
    counts = list(safe_word_counts.values())
    bars = ax4.bar(range(len(words)), counts, color='green', alpha=0.7)
    ax4.set_title('Top Words in Safe Emails', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Words')
    ax4.set_ylabel('Frequency')
    ax4.set_xticks(range(len(words)))
    ax4.set_xticklabels(words, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Word cloud analysis saved to: {output_path}")
    return output_path

def generate_combined_wordcloud(df, output_path="combined_wordcloud.png"):
    """Generate a single word cloud from all emails"""
    all_text = df['processed_text']
    
    plt.figure(figsize=(14, 8))
    wordcloud = WordCloud(
        width=1200, height=600,
        background_color='white',
        max_words=200,
        colormap='plasma',
        relative_scaling=0.5,
        random_state=42
    ).generate(' '.join(all_text.tolist()))
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Combined Email Text Word Cloud', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined word cloud saved to: {output_path}")
    return output_path

def main():
    """Main function to run word cloud analysis"""
    print("=== Email Word Cloud Generator ===")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Generate comparative analysis
    print("\nGenerating comparative word cloud analysis...")
    generate_comparative_wordclouds(df, "spam_vs_safe_wordclouds.png")
    
    # Generate combined word cloud
    print("\nGenerating combined word cloud...")
    generate_combined_wordcloud(df, "all_emails_wordcloud.png")
    
    # Print some statistics
    print("\n=== Text Statistics ===")
    print(f"Total emails: {len(df)}")
    print(f"Average text length: {df['processed_text'].str.len().mean():.1f} characters")
    print(f"Average word count: {df['processed_text'].str.split().str.len().mean():.1f} words")
    
    # Top words overall
    print("\n=== Top 10 Words Overall ===")
    top_words = get_top_words(df['processed_text'], n=10)
    for i, (word, count) in enumerate(top_words, 1):
        print(f"{i:2d}. {word:15s} ({count} times)")
    
    print("\nWord cloud generation completed!")

if __name__ == "__main__":
    main()
