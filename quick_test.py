import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Cấu hình
IMG_SIZE = 64
BATCH_SIZE = 32

# Đường dẫn dataset
train_path = 'face/train'
test_path = 'face/test'

# Các lớp cảm xúc
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def quick_test():
    """Test nhanh với một vài ảnh mẫu"""
    
    print("=== QUICK TEST ===")
    
    # Tạo test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    print(f"Found {test_generator.samples} test images")
    
    # Load model nếu có
    model_path = 'best_emotion_model.h5'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Test trên một batch
        test_generator.reset()
        batch = next(test_generator)
        images, labels = batch
        
        # Dự đoán
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        
        # Hiển thị kết quả
        print(f"\nTest results on {len(images)} images:")
        correct = 0
        for i in range(len(images)):
            predicted_emotion = emotion_classes[predicted_classes[i]]
            true_emotion = emotion_classes[true_classes[i]]
            confidence = np.max(predictions[i])
            
            if predicted_classes[i] == true_classes[i]:
                correct += 1
                status = "[OK]"
            else:
                status = "[NO]"
            
            print(f"{status} Image {i+1}: True={true_emotion}, Predicted={predicted_emotion} ({confidence:.2f})")
        
        accuracy = correct / len(images)
        print(f"\nQuick test accuracy: {accuracy:.2f} ({correct}/{len(images)})")
        
    else:
        print(f"Model not found at {model_path}")
        print("Please run train_model.py first to train the model")

if __name__ == "__main__":
    quick_test()
