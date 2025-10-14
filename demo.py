import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Các lớp cảm xúc
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def demo_with_sample_images():

    
    print("=== DEMO WITH SAMPLE IMAGES ===")
    
    # Load model
    model_path = 'best_emotion_model.h5'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please wait for training to complete or run train_model.py first")
        return
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Lấy một vài ảnh mẫu từ test set
    test_path = 'face/test'
    sample_images = []
    
    for emotion in emotion_classes:
        emotion_path = os.path.join(test_path, emotion)
        if os.path.exists(emotion_path):
            # Lấy ảnh đầu tiên
            files = [f for f in os.listdir(emotion_path) if f.endswith('.jpg')]
            if files:
                sample_images.append((os.path.join(emotion_path, files[0]), emotion))
    
    if not sample_images:
        print("No sample images found!")
        return
    
    print(f"Found {len(sample_images)} sample images")
    
    # Dự đoán cho từng ảnh
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (image_path, true_emotion) in enumerate(sample_images):
        if i >= 7:  # Chỉ hiển thị 7 ảnh
            break
            
        # Load và preprocess ảnh
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        img_input = np.expand_dims(img_input, axis=-1)
        
        # Dự đoán
        predictions = model.predict(img_input, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_emotion = emotion_classes[predicted_class]
        
        # Hiển thị ảnh
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_emotion}\nPredicted: {predicted_emotion}\nConfidence: {confidence:.2f}')
        axes[i].axis('off')
        
        print(f"Image {i+1}: True={true_emotion}, Predicted={predicted_emotion} ({confidence:.2f})")
    
    # Ẩn subplot thừa
    if len(sample_images) < 8:
        axes[7].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nDemo results saved to demo_results.png")

def demo_with_webcam():

    
    print("=== DEMO WITH WEBCAM ===")
    print("Press 'q' to quit")
    
    # Load model
    model_path = 'best_emotion_model.h5'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Crop face
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize và preprocess
            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict
            predictions = model.predict(face_roi, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            emotion = emotion_classes[predicted_class]
            
            # Vẽ rectangle và text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{emotion}: {confidence:.2f}', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Hiển thị frame
        cv2.imshow('Emotion Detection Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    print("=== EMOTION DETECTION DEMO ===")
    print("1. Demo with sample images")
    print("2. Demo with webcam")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        demo_with_sample_images()
    elif choice == '2':
        demo_with_webcam()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
