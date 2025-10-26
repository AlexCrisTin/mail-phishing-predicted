import customtkinter as ctk
from tkinter import messagebox
import joblib
import re
import numpy as np

#load
try:
    models = {
        "Random Forest": {
            "model": joblib.load('train/spam_classifier_model.pkl'),
            "vectorizer": joblib.load('train/spam_tfidf_vectorizer.pkl')
        },
        "Naive Bayes": {
            "model": joblib.load('train/spam_naive_bayes_model.pkl'),
            "vectorizer": joblib.load('train/spam_naive_bayes_vectorizer.pkl')
        },
        "Logistic Regression": {
            "model": joblib.load('train/spam_logistic_regression_model.pkl'),
            "vectorizer": joblib.load('train/spam_logistic_regression_vectorizer.pkl')
        }
    }
except FileNotFoundError as e:
    messagebox.showerror("Error", f"A model or vectorizer file was not found. Please ensure all models are trained.\nDetails: {e}")
    exit()

#predict
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def classify_email(content: str, model_name: str):
    if not content.strip():
        return "N/A", (0.0, 0.0)

    model_info = models[model_name]
    model = model_info["model"]
    vectorizer = model_info["vectorizer"]

    processed_text = preprocess_text(content)
    text_length = len(processed_text)
    word_count = len(processed_text.split())
    
    vectorized_text = vectorizer.transform([processed_text])
    
    features = np.hstack([vectorized_text.toarray(), [[text_length, word_count]]])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    result = "Phishing Detected" if prediction == 1 else "Email is Safe"
    return result, probabilities

class EmailClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(" ")
        self.geometry("800x750")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.grid_columnconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self):
        # Header
        header_frame = ctk.CTkFrame(self)
        header_frame.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            header_frame, text="Mail Checker",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white"
        ).pack(pady=20)

        # Input
        input_frame = ctk.CTkFrame(self, fg_color="#242424")
        input_frame.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(input_frame, text="Paste the email content:", font=ctk.CTkFont(size=16)).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        self.email_input = ctk.CTkTextbox(input_frame, height=150, corner_radius=6)
        self.email_input.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Control
        control_frame = ctk.CTkFrame(self, fg_color="#242424")
        control_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        control_frame.grid_columnconfigure(0, weight=1)
        self.analyze_button = ctk.CTkButton(control_frame, text="Analyze", command=self.on_classify, font=ctk.CTkFont(size=16, weight="bold"), height=40)
        self.analyze_button.pack(pady=10, padx=10, fill="x")

        # Results
        self.results_container = ctk.CTkFrame(self)
        self.results_container.grid(row=3, column=0, sticky="ew")
        self.results_container.grid_columnconfigure(0, weight=1)
        
        self.result_widgets = {}
        for i, model_name in enumerate(models.keys()):
            self._create_result_entry(model_name, i)
        
        self.results_container.grid_remove()

    def _create_result_entry(self, model_name, row_index):
        frame = ctk.CTkFrame(self.results_container)
        frame.grid(row=row_index, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        name_label = ctk.CTkLabel(frame, text=model_name, font=ctk.CTkFont(size=16, weight="bold"))
        name_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")
        
        result_label = ctk.CTkLabel(frame, text="Result: -", font=ctk.CTkFont(size=16))
        result_label.grid(row=1, column=0, padx=15, pady=10, sticky="w")

        safe_prob_label = ctk.CTkLabel(frame, text="Safe: -", font=ctk.CTkFont(size=14), text_color="#2ecc71")
        safe_prob_label.grid(row=0, column=1, padx=15, pady=(0, 10), sticky="e")

        phishing_prob_label = ctk.CTkLabel(frame, text="Phishing: -", font=ctk.CTkFont(size=14), text_color="#e74c3c")
        phishing_prob_label.grid(row=1, column=1, padx=15, pady=(0, 10), sticky="e")

        self.result_widgets[model_name] = {
            "frame": frame,
            "result": result_label,
            "safe": safe_prob_label,
            "phishing": phishing_prob_label
        }

    def on_classify(self):
        content = self.email_input.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showwarning("Input Error", "Please enter some email content to analyze.")
            return

        all_results = {}
        for model_name in models.keys():
            result, probabilities = classify_email(content, model_name)
            all_results[model_name] = (result, probabilities)
        
        self._update_all_results(all_results)
        self.results_container.grid()

    def _update_all_results(self, all_results):
        # Reset all frames to default
        for model_name in self.result_widgets:
            self.result_widgets[model_name]["frame"].configure(fg_color="#242424")

        # Determine consensus
        phishing_votes = sum(1 for result, _ in all_results.values() if result == "Phishing Detected")
        consensus = "Phishing Detected" if phishing_votes >= 2 else "Email is Safe"

        # Find best model for consensus
        best_model = None
        highest_prob = -1

        for model_name, (result, probabilities) in all_results.items():
            if result == consensus:
                prob_index = 1 if consensus == "Phishing Detected" else 0
                if probabilities[prob_index] > highest_prob:
                    highest_prob = probabilities[prob_index]
                    best_model = model_name
        
        # Update widgets and highlight best model
        for model_name, (result, probabilities) in all_results.items():
            widgets = self.result_widgets[model_name]
            
            color = "#2ecc71" if result == "Email is Safe" else "#e74c3c"
            widgets["result"].configure(text=f"Result: {result}", text_color=color)
            
            safe_prob, phishing_prob = probabilities
            widgets["safe"].configure(text=f"Safe: {safe_prob:.2%}")
            widgets["phishing"].configure(text=f"Phishing: {phishing_prob:.2%}")

            if model_name == best_model:
                widgets["frame"].configure(border_color="#E5FF00", border_width=2)

if __name__ == "__main__":
    app = EmailClassifierApp()
    app.mainloop()
