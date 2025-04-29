import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Data Preparation
data_dir = '/home/pavani-r/Documents/VSCODE/Datasets/Brain_MRI_dataset'
input_size = (128, 128)
batch_size = 32

# Updated Data Paths
train_dir = os.path.join(data_dir, 'training')
test_dir = os.path.join(data_dir, 'testing')

train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

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

# Build and Train Model
input_shape = (input_size[0], input_size[1], 3)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the model
model.save('resnet50_brain_tumor_model.h5')
print("ResNet50 model saved successfully.")

# Evaluate on the test set
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# Save accuracy in a text file
with open('resnet50_accuracy.txt', 'w') as f:
    f.write(f"{accuracy * 100:.2f}")

