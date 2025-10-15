import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import re
from io import BytesIO


st.set_page_config(page_title="📧 Email Spam/Phishing Report", page_icon="📧", layout="wide")
st.title("📧 Email Spam/Phishing Classification Dashboard")
st.caption("Dựa trên mô hình TF-IDF + RandomForest đã train từ `spam.csv`.")


@st.cache_resource(ttl=3600)
def load_artifacts():
    model = joblib.load("spam_classifier_model.pkl")
    vectorizer = joblib.load("spam_tfidf_vectorizer.pkl")
    return model, vectorizer


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text


@st.cache_data(ttl=600)
def load_base_data(max_rows_preview: int = 5000):
    """Load nhanh: chỉ đọc preview và tính phân phối nhãn theo chunks để tránh tải toàn bộ file."""
    usecols = ["Email Text", "Email Type"]
    try:
        # Preview nhanh cho hiển thị bảng
        df_preview = pd.read_csv("spam.csv", usecols=usecols, nrows=max_rows_preview)
        df_preview = df_preview.dropna(subset=usecols).copy()

        # Đếm phân phối nhãn theo từng khối (memory-efficient)
        label_counts = {}
        for chunk in pd.read_csv("spam.csv", usecols=usecols, chunksize=50000):
            vc = chunk["Email Type"].value_counts(dropna=True)
            for k, v in vc.items():
                label_counts[k] = label_counts.get(k, 0) + int(v)

        dist = (
            pd.DataFrame({"Loại": list(label_counts.keys()), "Số lượng": list(label_counts.values())})
            .sort_values("Số lượng", ascending=False)
            .reset_index(drop=True)
        )
        return df_preview, dist
    except Exception:
        return pd.DataFrame(columns=usecols), pd.DataFrame(columns=["Loại", "Số lượng"])  # Fallback


def predict_text(model, vectorizer, email_text: str) -> dict:
    processed = preprocess_text(email_text)
    text_length = len(processed)
    word_count = len(processed.split())
    X_text = vectorizer.transform([processed])
    X_other = np.array([[text_length, word_count]])
    X = np.hstack([X_text.toarray(), X_other])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = "Phishing Email" if pred == 1 else "Safe Email"
    return {
        "prediction": label,
        "phishing_probability": float(proba[1]),
        "safe_probability": float(proba[0]),
    }


def predict_dataframe(model, vectorizer, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df_proc = df.copy()
    df_proc[text_col] = df_proc[text_col].astype(str)
    df_proc["processed_text"] = df_proc[text_col].apply(preprocess_text)
    df_proc["text_length"] = df_proc["processed_text"].apply(len)
    df_proc["word_count"] = df_proc["processed_text"].apply(lambda x: len(x.split()))
    X_text = vectorizer.transform(df_proc["processed_text"])
    X_other = df_proc[["text_length", "word_count"]].values
    X = np.hstack([X_text.toarray(), X_other])
    preds = model.predict(X)
    probas = model.predict_proba(X)
    reverse_mapping = {0: "Safe Email", 1: "Phishing Email"}
    df_out = df_proc.copy()
    df_out["predicted_type"] = [reverse_mapping[p] for p in preds]
    df_out["phishing_probability"] = probas[:, 1]
    df_out["safe_probability"] = probas[:, 0]
    return df_out


tab_overview, tab_single, tab_batch, tab_reports = st.tabs([
    "📌 Tổng quan", "📝 Dự đoán 1 email", "📁 Dự đoán theo CSV", "📈 Báo cáo/Đồ thị",
])


with tab_overview:
    st.subheader("Tổng quan dữ liệu")
    base_df, dist = load_base_data()
    if not base_df.empty:
        st.write("Một phần dữ liệu gốc (spam.csv):")
        st.dataframe(base_df.head(20), use_container_width=True)
        # dist đã tính sẵn theo chunks để nhanh hơn
        chart = alt.Chart(dist).mark_bar().encode(
            x=alt.X("Loại:N", sort="-y"), y="Số lượng:Q", color="Loại:N", tooltip=["Loại", "Số lượng"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Không tìm thấy hoặc không đọc được spam.csv. Vui lòng đảm bảo file tồn tại.")


with tab_single:
    st.subheader("Dự đoán nội dung email đơn lẻ")
    model, vectorizer = load_artifacts()
    email_text = st.text_area("Nhập nội dung email", height=180)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("Phân loại"):
            if email_text.strip():
                res = predict_text(model, vectorizer, email_text)
                st.success(f"Kết quả: {res['prediction']}")
                st.metric("Xác suất Phishing", f"{res['phishing_probability']:.3f}")
                st.metric("Xác suất Safe", f"{res['safe_probability']:.3f}")
            else:
                st.warning("Vui lòng nhập nội dung email.")
    with col_b:
        st.caption("Nội dung đã chuẩn hoá (xem cách tiền xử lý):")
        if email_text.strip():
            st.code(preprocess_text(email_text))


with tab_batch:
    st.subheader("Dự đoán theo CSV")
    st.caption("CSV cần có cột 'Email Text'. Bạn có thể đổi tên cột bên dưới.")
    model, vectorizer = load_artifacts()
    uploaded = st.file_uploader("Tải lên CSV", type=["csv"])
    text_col = st.text_input("Tên cột văn bản", value="Email Text")
    if uploaded is not None:
        try:
            in_df = pd.read_csv(uploaded)
            if text_col not in in_df.columns:
                st.error(f"Không tìm thấy cột '{text_col}' trong file.")
            else:
                out_df = predict_dataframe(model, vectorizer, in_df, text_col=text_col)
                st.success("Đã phân loại xong.")
                st.dataframe(out_df.head(50), use_container_width=True)

                # Biểu đồ phân phối kết quả
                dist = out_df["predicted_type"].value_counts().reset_index()
                dist.columns = ["Loại", "Số lượng"]
                chart = alt.Chart(dist).mark_bar().encode(
                    x=alt.X("Loại:N", sort="-y"), y="Số lượng:Q", color="Loại:N", tooltip=["Loại", "Số lượng"]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

                # Download kết quả
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Tải kết quả CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.exception(e)


with tab_reports:
    st.subheader("Báo cáo/Đồ thị")
    st.caption("Hình tổng hợp: Learning curve, Validation curve, ROC, Confusion matrix")
    try:
        st.image("report_plots.png", caption="Model Diagnostics", use_column_width=True)
    except Exception:
        st.info("Không tìm thấy 'report_plots.png'. Hãy chạy: python plot_report.py")

