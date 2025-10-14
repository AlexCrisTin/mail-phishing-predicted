import json
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

def monitor_training():
    """Monitor training progress"""
    
    print("=== TRAINING MONITOR ===")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Kiểm tra xem có file model nào được tạo chưa
            model_files = ['best_emotion_model.h5', 'emotion_model_final.h5']
            results_file = 'training_results.json'
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking training progress...")
            
            # Kiểm tra model files
            for model_file in model_files:
                if os.path.exists(model_file):
                    size = os.path.getsize(model_file) / (1024*1024)  # MB
                    print(f"✓ {model_file} exists ({size:.1f} MB)")
                else:
                    print(f"✗ {model_file} not found")
            
            # Kiểm tra results file
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    print(f"✓ Training completed!")
                    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
                    print(f"  Test Loss: {results['test_loss']:.4f}")
                    print(f"  Epochs trained: {results['epochs_trained']}")
                    break
                except:
                    print(f"✗ {results_file} exists but cannot read")
            else:
                print(f"✗ {results_file} not found - training still in progress")
            
            # Chờ 30 giây trước khi kiểm tra lại
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def check_dataset():
    """Kiểm tra dataset"""
    
    print("=== DATASET CHECK ===")
    
    train_path = 'face/train'
    test_path = 'face/test'
    
    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    print("Training set:")
    total_train = 0
    for emotion in emotion_classes:
        emotion_path = os.path.join(train_path, emotion)
        if os.path.exists(emotion_path):
            count = len([f for f in os.listdir(emotion_path) if f.endswith('.jpg')])
            total_train += count
            print(f"  {emotion}: {count} images")
        else:
            print(f"  {emotion}: NOT FOUND")
    
    print(f"  Total training: {total_train} images")
    
    print("\nTest set:")
    total_test = 0
    for emotion in emotion_classes:
        emotion_path = os.path.join(test_path, emotion)
        if os.path.exists(emotion_path):
            count = len([f for f in os.listdir(emotion_path) if f.endswith('.jpg')])
            total_test += count
            print(f"  {emotion}: {count} images")
        else:
            print(f"  {emotion}: NOT FOUND")
    
    print(f"  Total test: {total_test} images")
    print(f"  Total dataset: {total_train + total_test} images")

def main():
    """Main function"""
    print("Choose an option:")
    print("1. Check dataset")
    print("2. Monitor training")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        check_dataset()
    elif choice == '2':
        monitor_training()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
