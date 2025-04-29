import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D

# === DATA PREPARATION ===
data_dir = '/home/pavani-r/Documents/VSCODE/Datasets/Brain_MRI_dataset'
input_size = (128, 128)
batch_size = 32

# Directory paths
train_dir = os.path.join(data_dir, 'training')
test_dir = os.path.join(data_dir, 'testing')

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

input_shape = (input_size[0], input_size[1], 3)

# === MobileNetV2 MODEL ===
print("\nTraining MobileNetV2...")
base_model_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

model_mobilenet = Sequential([
    base_model_mobilenet,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model_mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_mobilenet.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

model_mobilenet.save('mobilenetv2_brain_tumor_model.h5')
print("MobileNetV2 model saved successfully.")

# Evaluate MobileNetV2
loss_mnet, acc_mnet = model_mobilenet.evaluate(test_generator)
print(f"MobileNetV2 Test Accuracy: {acc_mnet * 100:.2f}%")

with open('mobilenetv2_accuracy.txt', 'w') as f:
    f.write(f"{acc_mnet * 100:.2f}")
