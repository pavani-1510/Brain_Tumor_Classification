import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization

# === DATA PREPARATION ===
data_dir = '/home/pavani-r/Documents/VSCODE/Datasets/Brain_MRI_dataset'
input_size = (128, 128)
batch_size = 32

train_dir = os.path.join(data_dir, 'training')
test_dir = os.path.join(data_dir, 'testing')

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

# === AUTOENCODER ===
input_img = Input(shape=input_shape)

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder (optional: only if you want to reconstruct)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Encoder model for classification
encoder_model = Model(inputs=input_img, outputs=encoded)

# Classification Head
x = GlobalAveragePooling2D()(encoded)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

# Final model
autoencoder_classifier = Model(inputs=input_img, outputs=output)

# Compile and train
autoencoder_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining Autoencoder-based classifier...")
autoencoder_classifier.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

autoencoder_classifier.save('autoencoder_brain_tumor_model.h5')
print("Autoencoder model saved successfully.")

# Evaluate
loss_ae, acc_ae = autoencoder_classifier.evaluate(test_generator)
print(f"Autoencoder Test Accuracy: {acc_ae * 100:.2f}%")

with open('autoencoder_accuracy.txt', 'w') as f:
    f.write(f"{acc_ae * 100:.2f}")
