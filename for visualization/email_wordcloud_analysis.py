import re
import logging
from collections import Counter
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

#config
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

#data processing
def preprocess_text(text: str) -> str:
    """Clean up"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())


def load_data(csv_path: str = "spam.csv") -> Optional[pd.DataFrame]:
    """Load and preprocess csv"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"Can't find: {csv_path}")
        return None
    logging.info(f"Loaded: {len(df)} row, col: {list(df.columns)}")

    if not {"Email Text", "Email Type"}.issubset(df.columns):
        raise ValueError("CSV need col 'Email Text' and 'Email Type'.")

    df = df.dropna(subset=["Email Text", "Email Type"]).copy()
    df["processed_text"] = df["Email Text"].astype(str).apply(preprocess_text)
    df = df[df["processed_text"].str.len() > 0].copy()

    logging.info(f"After cleaing: {len(df)} dòng")
    logging.info(f"Type: {df['Email Type'].value_counts().to_dict()}")
    return df


#Ultilities
def get_top_words(text_data, n: int = 20) -> list[Tuple[str, int]]:
    """N popular words"""
    combined_text = ' '.join(text_data.tolist()) if isinstance(text_data, pd.Series) else ' '.join(text_data)
    counts = Counter(w for w in combined_text.split() if len(w) > 2)
    return counts.most_common(n)


def plot_word_frequency(word_counts: list[Tuple[str, int]], title: str = "Top Words", n: int = 20):
    """word frequency"""
    top_words = dict(word_counts[:n])
    plt.figure(figsize=(12, 8))
    plt.bar(top_words.keys(), top_words.values(), color='skyblue', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()


#word cloud
def create_wordcloud(text_data, title: str, colormap: str, output_path: Optional[str] = None):
    text = ' '.join(text_data.tolist()) if isinstance(text_data, pd.Series) else ' '.join(text_data)
    wordcloud = WordCloud(
        width=800, height=400, background_color='white',
        max_words=100, colormap=colormap, random_state=42
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {output_path}")
    else:
        plt.show()


def generate_comparative_wordclouds(df: pd.DataFrame, output_path: str = "wordcloud_analysis.png"):
    """Compare"""
    spam = df[df['Email Type'] == 'Phishing Email']['processed_text']
    safe = df[df['Email Type'] == 'Safe Email']['processed_text']

    logging.info(f"Spam count: {len(spam)}, Safe count: {len(safe)}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Word Cloud Email', fontsize=16, fontweight='bold')

    # WordCloud Spam
    for (ax, data, cmap, title) in [
        (axes[0, 0], spam, 'Reds', 'Phishing Emails'),
        (axes[0, 1], safe, 'Greens', 'Safe Emails')
    ]:
        wc = WordCloud(width=400, height=300, background_color='white',
                       max_words=100, colormap=cmap, random_state=42).generate(' '.join(data))
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Word Frequency
    for (ax, data, color, title) in [
        (axes[1, 0], spam, 'red', 'Từ phổ biến trong Spam mail'),
        (axes[1, 1], safe, 'green', 'Từ phổ biến trong Safe mail')
    ]:
        top_words = get_top_words(data, 15)
        words, counts = zip(*top_words)
        bars = ax.bar(words, counts, color=color, alpha=0.7)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, count + 0.1, str(count),
                    ha='center', va='bottom', fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved wordcloud: {output_path}")


def generate_combined_wordcloud(df: pd.DataFrame, output_path: str = "combined_wordcloud.png"):
    """Word cloud for all"""
    all_text = ' '.join(df['processed_text'].tolist())
    wc = WordCloud(
        width=1200, height=600, background_color='white',
        max_words=200, colormap='plasma', random_state=42
    ).generate(all_text)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Combined Email Word Cloud', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


# main
def main():
    logging.info("=== Word Cloud Generator ===")

    df = load_data()
    if df is None:
        return

    generate_comparative_wordclouds(df, "spam_vs_safe_wordclouds.png")
    generate_combined_wordcloud(df, "all_emails_wordcloud.png")

    logging.info("=== Thống kê ===")
    logging.info(f"Tổng số email: {len(df)}")
    logging.info(f"Độ dài trung bình: {df['processed_text'].str.len().mean():.1f} ký tự")
    logging.info(f"Số từ trung bình: {df['processed_text'].str.split().str.len().mean():.1f}")

    logging.info("=== Top 10 ===")
    for i, (w, c) in enumerate(get_top_words(df['processed_text'], 10), 1):
        logging.info(f"{i:2d}. {w:15s} ({c})")

    logging.info("Done")


if __name__ == "__main__":
    main()
