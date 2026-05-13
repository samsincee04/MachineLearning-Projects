from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

TEST_DIR = Path("dataset/processed/test")
IMG_SIZE = (48, 48)
CLASS_NAMES = ["no_shoe", "shoe"]


if __name__ == "__main__":
    model_path = Path("project1_model.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found. Run `python train.py` first.")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=False,
    )

    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    model = keras.models.load_model(model_path)

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy on processed/test: {acc:.4f}")

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_label = int(preds[0][0] >= 0.5)
        true_label = int(labels.numpy()[0][0])

        y_true.append(true_label)
        y_pred.append(pred_label)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
