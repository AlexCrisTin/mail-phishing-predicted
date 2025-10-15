import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split, learning_curve, validation_curve, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_auc_score, roc_curve


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text


def load_data(csv_path: str = "spam.csv"):
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
    return X, y


def build_pipeline():
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, min_samples_split=5, min_samples_leaf=2)),
        ]
    )


def plot_learning_curve(ax, estimator, X, y, title: str = "Learning Curve"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    ax.set_title(title)
    ax.plot(train_sizes, train_mean, "o-", color="#1f77b4", label="Training F1")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")
    ax.plot(train_sizes, valid_mean, "o-", color="#ff7f0e", label="Validation F1")
    ax.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.15, color="#ff7f0e")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 score")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_validation_curve(ax, X, y, param_name: str = "clf__max_depth", param_range=None, title: str = "Validation Curve"):
    if param_range is None:
        param_range = [5, 10, 15, 20, 25, 30]
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_scores, valid_scores = validation_curve(
        estimator=pipeline,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    ax.set_title(title + f" ({param_name})")
    ax.plot(param_range, train_mean, "o-", color="#1f77b4", label="Training F1")
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")
    ax.plot(param_range, valid_mean, "o-", color="#ff7f0e", label="Validation F1")
    ax.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, alpha=0.15, color="#ff7f0e")
    ax.set_xlabel(param_name)
    ax.set_ylabel("F1 score")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_roc_and_confusion(ax_roc, ax_cm, X, y):
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    ax_roc.plot(fpr, tpr, color="#d62728", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc="lower right")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion Matrix (Hold-out)")


def generate_report(output_path: str = "report_plots.png"):
    X, y = load_data("spam.csv")
    pipeline = build_pipeline()

    plt.figure(figsize=(14, 10))
    gs = plt.GridSpec(2, 2, hspace=0.25, wspace=0.25)

    ax1 = plt.subplot(gs[0, 0])
    plot_learning_curve(ax1, pipeline, X, y, title="Learning Curve (Over/Underfitting)")

    ax2 = plt.subplot(gs[0, 1])
    plot_validation_curve(ax2, X, y, param_name="clf__max_depth", param_range=[5, 10, 15, 20, 25, 30], title="Validation Curve")

    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    plot_roc_and_confusion(ax3, ax4, X, y)

    plt.suptitle("Email Spam/Phishing Classification - Model Diagnostics", fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return output_path


if __name__ == "__main__":
    out = generate_report("report_plots.png")
    print(f"Saved report image to {out}")