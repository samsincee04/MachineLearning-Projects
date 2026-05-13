from pathlib import Path
import tensorflow as tf
import numpy as np

MODEL_PATH = Path("project1_model.h5")
TFLITE_PATH = Path("project1_model_int8.tflite")
TRAIN_DIR = Path("dataset/processed/train")
IMG_SIZE = (48, 48)
BATCH_SIZE = 1
CLASS_NAMES = ["no_shoe", "shoe"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Representative dataset for full integer quantization
rep_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

def representative_dataset():
    for images, _ in rep_ds.take(20):
        images = tf.cast(images, tf.float32) / 255.0
        yield [images]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Saved quantized TFLite model to {TFLITE_PATH} ({len(tflite_model)} bytes)")
