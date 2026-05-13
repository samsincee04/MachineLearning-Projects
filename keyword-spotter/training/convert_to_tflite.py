import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

# Default: last training run. For deployment, use `frozen_runs/.../kws_model.keras`
# (see freeze_passing_run.py and training/frozen_runs/current_candidate.json).
_DEFAULT_KERAS = _SCRIPT_DIR / "kws_model.keras"
REP_DATA_PATH = _SCRIPT_DIR / "rep_data.npy"

TFLITE_FLOAT_PATH = _SCRIPT_DIR / "kws_model_float.tflite"
TFLITE_DYNAMIC_PATH = _SCRIPT_DIR / "kws_model_dynamic.tflite"
TFLITE_INT8_PATH = _SCRIPT_DIR / "kws_model_int8.tflite"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert kws Keras to float, dynamic, and int8 TFLite (writes to training/)."
    )
    p.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_KERAS,
        help=f"Keras .keras file (default: {_DEFAULT_KERAS.name} in this directory).",
    )
    return p.parse_args()


args = _parse_args()
model_path = args.model
if not model_path.is_absolute():
    cand_cwd = (Path.cwd() / model_path).resolve()
    cand_script = (_SCRIPT_DIR / model_path).resolve()
    if cand_cwd.is_file():
        model_path = cand_cwd
    elif cand_script.is_file():
        model_path = cand_script
    else:
        raise SystemExit(
            f"Model not found: {args.model}\n  Tried: {cand_cwd}\n  Tried: {cand_script}"
        )
else:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

print("Loading Keras model...")
print(f"  {model_path}")
model = tf.keras.models.load_model(model_path)

print("\nConverting float TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter.convert()
Path(TFLITE_FLOAT_PATH).write_bytes(tflite_float)
print(f"Saved float TFLite model to {TFLITE_FLOAT_PATH}")

print("\nConverting dynamic-range TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic = converter.convert()
Path(TFLITE_DYNAMIC_PATH).write_bytes(tflite_dynamic)
print(f"Saved dynamic-range TFLite model to {TFLITE_DYNAMIC_PATH}")

print("\nLoading representative data for full INT8 quantization...")
if not REP_DATA_PATH.is_file():
    raise SystemExit(f"Missing {REP_DATA_PATH}; run speech_training.py first to create rep_data.npy.")
rep_data = np.load(REP_DATA_PATH)
print(f"Representative data shape: {rep_data.shape}, dtype: {rep_data.dtype}")
print(f"Representative data stats: min={rep_data.min()}, max={rep_data.max()}, mean={rep_data.mean()}")

def representative_dataset():
    for i in range(min(100, len(rep_data))):
        sample = rep_data[i:i+1].astype(np.float32)
        yield [sample]

print("\nConverting full INT8 TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
Path(TFLITE_INT8_PATH).write_bytes(tflite_int8)
print(f"Saved full INT8 TFLite model to {TFLITE_INT8_PATH}")

print("\nDone.")