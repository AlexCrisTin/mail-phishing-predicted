import customtkinter as ctk
from tkinter import messagebox
import joblib
import re
import numpy as np


try:
    model = joblib.load('train/spam_classifier_model.pkl')
    vectorizer = joblib.load('train/spam_tfidf_vectorizer.pkl')
except FileNotFoundError:
    messagebox.showerror("Error", "Model or vectorizer not found. Please train the model first.")
    exit()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def classify_email(content: str):
    if not content.strip():
        return "N/A", (0.0, 0.0)

    processed_text = preprocess_text(content)
    text_length = len(processed_text)
    word_count = len(processed_text.split())
    
    vectorized_text = vectorizer.transform([processed_text])
    
    features = np.hstack([vectorized_text.toarray(), [[text_length, word_count]]])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    if prediction == 1:
        return "Phishing Detected", probabilities
    else:
        return "Email is Safe", probabilities

class EmailClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.grid_columnconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self):
        # Header
        header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#2c3e50")
        header_frame.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
        header_frame, text="Email Checker",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white"
        ).pack(pady=20)

        # Input
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            input_frame, text="Paste the email content below:",
            font=ctk.CTkFont(size=16)
        ).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.email_input = ctk.CTkTextbox(input_frame, height=200, corner_radius=6)
        self.email_input.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Control 
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        control_frame.grid_columnconfigure(0, weight=1)

        self.analyze_button = ctk.CTkButton(
            control_frame, text="Analyze Email",
            command=self.on_classify,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40
        )
        self.analyze_button.pack(pady=10, padx=10, fill="x")

        # Result 
        self.result_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#34495e")
        self.result_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_remove()

        self.result_label = ctk.CTkLabel(
            self.result_frame, text="",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.result_label.grid(row=0, column=0, columnspan=2, pady=(15, 10))

        # Probability 
        self.safe_prob_label = ctk.CTkLabel(
            self.result_frame, text="Safe: 0.00%",
            font=ctk.CTkFont(size=16),
            text_color="#2ecc71"
        )
        self.safe_prob_label.grid(row=1, column=0, pady=(0, 15), padx=(10, 5), sticky="e")

        self.phishing_prob_label = ctk.CTkLabel(
            self.result_frame, text="Phishing: 0.00%",
            font=ctk.CTkFont(size=16),
            text_color="#e74c3c"
        )
        self.phishing_prob_label.grid(row=1, column=1, pady=(0, 15), padx=(5, 10), sticky="w")


    def on_classify(self):
        content = self.email_input.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showwarning("Input Error", "Please enter some email content to analyze.")
            return

        result, probabilities = classify_email(content)
        self._update_result(result, probabilities)
        self.result_frame.grid()

    def _update_result(self, result, probabilities):
        if result == "Phishing Detected":
            self.result_label.configure(text=result, text_color="#e74c3c")
        else:
            self.result_label.configure(text=result, text_color="#2ecc71")
        
        safe_prob, phishing_prob = probabilities
        self.safe_prob_label.configure(text=f"Safe: {safe_prob:.2%}")
        self.phishing_prob_label.configure(text=f"Phishing: {phishing_prob:.2%}")

if __name__ == "__main__":
    app = EmailClassifierApp()
    app.mainloop()
