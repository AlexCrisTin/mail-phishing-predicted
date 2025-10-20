import customtkinter as ctk
from tkinter import messagebox 

# Định nghĩa các ngưỡng (thresholds)
TOXIC_THRESHOLD_HIGH = 0.60
TOXIC_THRESHOLD_LOW = 0.40

# Chuỗi mẫu cho nội dung Cấm và Nghi ngờ
FORBIDDEN_PHRASE = "You are a [Cấm: Xúc phạm mẫu] person."
SUSPICIOUS_PHRASE = "Please click here to update your subscription details." # Nội dung có thể nghi ngờ

# --- Hàm mô phỏng phân loại ---
def classify_email(email_content):
    """Mô phỏng hàm phân loại, nhận diện mức độ Toxic: Safe, Review, hoặc Toxic."""
    standardized_content = email_content.lower()
    
    # 1. Trường hợp Toxic rõ ràng
    if "[cấm: xúc phạm mẫu]" in standardized_content:
        toxic_prob = 0.920
        safe_prob = 0.080
    # 2. Trường hợp Nghi ngờ (Suspicious)
    elif "click here" in standardized_content or "update subscription" in standardized_content:
        # Giả định xác suất rơi vào vùng nghi ngờ (ví dụ: 55%)
        toxic_prob = 0.550 
        safe_prob = 0.450
    # 3. Trường hợp Safe mặc định
    else:
        toxic_prob = 0.050
        safe_prob = 0.950

    # Xác định trạng thái cuối cùng dựa trên xác suất
    if toxic_prob >= TOXIC_THRESHOLD_HIGH:
        result = "Toxic Email"
    elif toxic_prob > TOXIC_THRESHOLD_LOW and toxic_prob < TOXIC_THRESHOLD_HIGH:
        result = "Review Required" # Trạng thái mới: Cần xem xét sau
    else:
        result = "Safe Email"

    return standardized_content, result, toxic_prob, safe_prob

# --- Hàm xử lý sự kiện nút "Phân loại" ---
def classification_event():
    """Lấy nội dung, phân loại và cập nhật giao diện. Thêm logic chặn/xem xét."""
    email_content = email_input_textbox.get("1.0", "end-1c")

    if not email_content.strip():
        standardized_content = ""
        result = "N/A"
        toxic_prob = 0.000
        safe_prob = 0.000
    else:
        standardized_content, result, toxic_prob, safe_prob = classify_email(email_content)

        # --- LOGIC TỰ ĐỘNG CHẶN/XEM XÉT ---
        if result == "Toxic Email" and auto_block_var.get() == 1:
            messagebox.showwarning(
                "CẢNH BÁO: ĐÃ TỰ ĐỘNG CHẶN",
                f"Hệ thống đã tự động chặn email Toxic. Xác suất Toxic: {toxic_prob:.3f}"
            )
        elif result == "Review Required":
            # Thông báo chuyển vào khu vực xem xét
            messagebox.showinfo(
                "THÔNG BÁO: CHUYỂN VÀO XEM XÉT",
                f"Email được đánh dấu Nghi ngờ và đã chuyển vào thư mục 'Xem xét sau'. Xác suất Toxic: {toxic_prob:.3f}"
            )
        # ---------------------------------

    # Cập nhật Nội dung đã chuẩn hóa
    standardized_textbox.configure(state="normal")
    standardized_textbox.delete("1.0", "end")
    standardized_textbox.insert("1.0", standardized_content)
    standardized_textbox.configure(state="disabled")

    # Cập nhật kết quả phân loại và màu sắc
    if result == "Safe Email":
        result_label.configure(text=f"Kết quả: {result}", fg_color="#2e7d32") # Xanh lá
    elif result == "Toxic Email":
        result_label.configure(text=f"Kết quả: {result}", fg_color="#b80000") # Đỏ cho Toxic
    elif result == "Review Required":
        result_label.configure(text=f"Kết quả: {result}", fg_color="#3a86ff") # Xanh dương cho Nghi ngờ/Xem xét
    else:
        result_label.configure(text=f"Kết quả: {result}", fg_color="#434343")


    # Cập nhật xác suất
    toxic_prob_value.configure(text=f"{toxic_prob:.3f}")
    safe_prob_value.configure(text=f"{safe_prob:.3f}")


# --- Thiết lập cửa sổ chính ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue") 

app = ctk.CTk()
app.title("Dự đoán nội dung email (Có mức độ Nghi ngờ)")
app.geometry("700x800") 
app.grid_columnconfigure(0, weight=1)

# --- 1. Tiêu đề ---
title_label = ctk.CTkLabel(app, 
                           text="Dự đoán nội dung email đơn lẻ", 
                           font=ctk.CTkFont(size=24, weight="bold"))
title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

# --- 2. Ô nhập nội dung email ---
input_label = ctk.CTkLabel(app, text="Nhập nội dung email", font=ctk.CTkFont(size=16))
input_label.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="w")

email_input_textbox = ctk.CTkTextbox(app, height=150, activate_scrollbars=True)
email_input_textbox.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
# Thiết lập nội dung mặc định là trường hợp NGHI NGỜ
email_input_textbox.insert("1.0", SUSPICIOUS_PHRASE) 

# --- 3. Khung chứa nút Phân loại, Tự động Chặn và Nội dung đã chuẩn hóa ---
input_frame = ctk.CTkFrame(app, fg_color="transparent")
input_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
input_frame.grid_columnconfigure(1, weight=1) 

# Khung điều khiển
control_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
control_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nw")

classify_button = ctk.CTkButton(control_frame, 
                                text="Phân loại", 
                                command=classification_event,
                                fg_color="#434343")
classify_button.grid(row=0, column=0, padx=0, pady=(0, 10), sticky="ew")

auto_block_var = ctk.IntVar(value=0) # Mặc định TẮT
auto_block_checkbox = ctk.CTkCheckBox(control_frame, 
                                      text="Tự động chặn mail Toxic", 
                                      variable=auto_block_var,
                                      checkbox_width=20,
                                      checkbox_height=20)
auto_block_checkbox.grid(row=1, column=0, padx=0, pady=(10, 0), sticky="w")

# Nội dung đã chuẩn hóa (Cột 2)
standardized_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
standardized_frame.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="ew")

standardized_label = ctk.CTkLabel(standardized_frame, 
                                  text="Nội dung đã chuẩn hóa (xem cách tiền xử lý):", 
                                  font=ctk.CTkFont(size=14))
standardized_label.grid(row=0, column=0, padx=0, pady=(0, 5), sticky="w")

standardized_textbox = ctk.CTkTextbox(standardized_frame, 
                                      height=40, 
                                      state="normal", 
                                      text_color="white",
                                      fg_color="#343638", 
                                      border_color="#343638")
standardized_textbox.grid(row=1, column=0, padx=0, pady=0, sticky="ew")
standardized_textbox.insert("1.0", SUSPICIOUS_PHRASE.lower()) # Nội dung mặc định
standardized_textbox.configure(state="disabled") 

# --- 4. Kết quả Phân loại (Hộp màu) ---
result_label = ctk.CTkLabel(app, 
                            text="Kết quả: Review Required", # Kết quả mặc định: Nghi ngờ
                            font=ctk.CTkFont(size=18, weight="bold"),
                            fg_color="#3a86ff", # Màu Xanh Dương cho Xem xét
                            corner_radius=6,
                            height=40)
result_label.grid(row=4, column=0, padx=20, pady=(20, 30), sticky="w")

# --- 5. Xác suất Toxic ---
toxic_title = ctk.CTkLabel(app, text="Xác suất Toxic", font=ctk.CTkFont(size=16))
toxic_title.grid(row=5, column=0, padx=20, pady=(5, 0), sticky="w")

toxic_prob_value = ctk.CTkLabel(app, 
                                   text="0.550", # Xác suất ở mức nghi ngờ
                                   font=ctk.CTkFont(size=36, weight="normal"))
toxic_prob_value.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="w")

# --- 6. Xác suất Safe ---
safe_title = ctk.CTkLabel(app, text="Xác suất Safe", font=ctk.CTkFont(size=16))
safe_title.grid(row=7, column=0, padx=20, pady=(5, 0), sticky="w")

safe_prob_value = ctk.CTkLabel(app, 
                               text="0.450", # Xác suất ở mức nghi ngờ
                               font=ctk.CTkFont(size=36, weight="normal"))
safe_prob_value.grid(row=8, column=0, padx=20, pady=(0, 20), sticky="w")

# Chạy ứng dụng
app.mainloop()
