import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add, Reshape, Permute
)

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

# === TRANSFORMER BLOCK ===
def transformer_block(inputs, num_heads=4, ff_dim=128):
    # Layer Norm 1 + Multi-head Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(x, x)
    x = Add()([inputs, attention_output])

    # Layer Norm 2 + Feed Forward
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(ff_dim, activation='relu')(y)
    y = Dense(inputs.shape[-1])(y)
    return Add()([x, y])


# === CNN + Transformer MODEL ===
print("\nTraining CNN + Transformer...")

inputs = Input(shape=(input_size[0], input_size[1], 3))

# CNN Feature Extractor
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Reshape feature map to sequence of patches
# From (batch, H, W, C) → (batch, H*W, C)
shape = x.shape
patches = Reshape((shape[1] * shape[2], shape[3]))(x)

# Transformer Encoder Layer
x = transformer_block(patches)

# Pooling and Output
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(train_generator.num_classes, activation='softmax')(x)

model_transformer = Model(inputs=inputs, outputs=outputs)

model_transformer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_transformer.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

model_transformer.save('cnn_transformer_brain_tumor_model.h5')
print("CNN + Transformer model saved successfully.")

# Evaluate
loss_trans, acc_trans = model_transformer.evaluate(test_generator)
print(f"CNN + Transformer Test Accuracy: {acc_trans * 100:.2f}%")

with open('cnn_transformer_accuracy.txt', 'w') as f:
    f.write(f"{acc_trans * 100:.2f}")
