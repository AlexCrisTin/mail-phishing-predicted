import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import json

# Cấu hình cải tiến
IMG_SIZE = 96  # Tăng kích thước ảnh lên 96x96
BATCH_SIZE = 32
EPOCHS = 100  # Tăng số epochs
LEARNING_RATE = 0.001

# Đường dẫn dataset
train_path = 'face/train'
test_path = 'face/test'

# Các lớp cảm xúc
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(emotion_classes)

print(f"So luong lop cam xuc: {num_classes}")
print(f"Cac lop: {emotion_classes}")

# Data augmentation cải tiến
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Tăng rotation
    width_shift_range=0.3,  # Tăng shift
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3,  # Tăng zoom
    shear_range=0.3,  # Tăng shear
    brightness_range=[0.7, 1.3],  # Thêm brightness
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'
)

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

print(f"So luong anh training: {train_generator.samples}")
print(f"So luong anh validation: {validation_generator.samples}")
print(f"So luong anh test: {test_generator.samples}")

# Tạo model CNN cải tiến
def create_improved_emotion_model():
    model = Sequential([
        # Block 1 - Tăng số filters
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1),
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 4 - Thêm block mới
        Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Global Average Pooling thay vì Flatten
        GlobalAveragePooling2D(),
        
        # Dense layers cải tiến
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Learning rate scheduler
def lr_schedule(epoch):
    """Learning rate scheduler"""
    initial_lr = 0.001
    if epoch < 20:
        return initial_lr
    elif epoch < 40:
        return initial_lr * 0.5
    elif epoch < 60:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01

# Tạo model
model = create_improved_emotion_model()

# Compile với optimizer cải tiến
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_3_accuracy']
)

# In thông tin model
model.summary()

# Callbacks cải tiến
callbacks = [
    ModelCheckpoint(
        'best_improved_emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # Tăng patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,  # Tăng patience
        min_lr=1e-7,
        verbose=1
    ),
    LearningRateScheduler(lr_schedule, verbose=1)
]

print("Bat dau training improved model...")

# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("Training hoan thanh!")

# Lưu model cuối cùng
model.save('improved_emotion_model_final.h5')
print("Improved model da duoc luu!")

# Vẽ biểu đồ training history
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-3 Accuracy
    if 'top_3_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
        axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Vẽ biểu đồ
plot_training_history(history)

# Đánh giá model trên test set
print("\nDanh gia improved model tren test set...")
test_loss, test_accuracy, test_top3_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Top-3 Accuracy: {test_top3_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Dự đoán trên test set
print("\nTao confusion matrix...")
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=emotion_classes, yticklabels=emotion_classes)
plt.title('Improved Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification Report
print("\nImproved Model Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=emotion_classes))

# Lưu kết quả
results = {
    'test_accuracy': float(test_accuracy),
    'test_top3_accuracy': float(test_top3_accuracy),
    'test_loss': float(test_loss),
    'emotion_classes': emotion_classes,
    'model_architecture': 'Improved CNN with 4 Conv blocks + GlobalAveragePooling + Dense layers',
    'img_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'epochs_trained': len(history.history['accuracy']),
    'improvements': [
        'Increased image size to 96x96',
        'Added more Conv blocks (4 blocks)',
        'Increased filters (64, 128, 256, 512)',
        'Added L2 regularization',
        'Used GlobalAveragePooling instead of Flatten',
        'Added more Dense layers',
        'Improved data augmentation',
        'Added learning rate scheduling',
        'Increased epochs to 100'
    ]
}

with open('improved_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nKet qua improved model da duoc luu vao improved_training_results.json")
print(f"Improved model tot nhat da duoc luu vao best_improved_emotion_model.h5")
print(f"Improved model cuoi cung da duoc luu vao improved_emotion_model_final.h5")
