import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Dataset folder path
DATASET_DIR = "."

IMG_SIZE = 128
BATCH_SIZE = 32

# Load images from train and train1 folders
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_names=["train", "train1"]
)

# Normalize images
normalization_layer = layers.Rescaling(1.0 / 255)

dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

# Split dataset
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

# Create CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save model
model.save("mask_model.keras")

print("Model saved as mask_model.keras")