import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Data Preparation
data_dir = '/home/pavani-r/Documents/VSCODE/DL/Dataset'
input_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build and Train VGG16 Model
input_shape = (input_size[0], input_size[1], 3)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

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
    validation_data=validation_generator
)

# Save the model
model.save('vgg16_brain_tumor_model.h5')
print("VGG16 model saved successfully.")