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
                            text="Kết quả: Review Required", 
                            font=ctk.CTkFont(size=18, weight="bold"),
                            fg_color="#3a86ff", 
                            corner_radius=6,
                            height=40)
result_label.grid(row=4, column=0, padx=20, pady=(20, 30), sticky="w")

# --- 5. Xác suất Toxic ---
toxic_title = ctk.CTkLabel(app, text="Xác suất Toxic", font=ctk.CTkFont(size=16))
toxic_title.grid(row=5, column=0, padx=20, pady=(5, 0), sticky="w")

toxic_prob_value = ctk.CTkLabel(app, 
                                   text="0.550", 
                                   font=ctk.CTkFont(size=36, weight="normal"))
toxic_prob_value.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="w")

# --- 6. Xác suất Safe ---
safe_title = ctk.CTkLabel(app, text="Xác suất Safe", font=ctk.CTkFont(size=16))
safe_title.grid(row=7, column=0, padx=20, pady=(5, 0), sticky="w")

safe_prob_value = ctk.CTkLabel(app, 
                               text="0.450", 
                               font=ctk.CTkFont(size=36, weight="normal"))
safe_prob_value.grid(row=8, column=0, padx=20, pady=(0, 20), sticky="w")

app.mainloop()
