from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TRAIN_DIR = Path("dataset/processed/train")
VAL_DIR = Path("dataset/processed/val")
IMG_SIZE = (48, 48)
BATCH_SIZE = 8
EPOCHS = 30

CLASS_NAMES = ["no_shoe", "shoe"]

def build_model(input_shape) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),

            layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
if __name__ == "__main__":
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    model.save("project1_model.h5")
    print("Saved model to project1_model.h5")

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
