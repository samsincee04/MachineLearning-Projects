import numpy as np
import tensorflow as tf
from pathlib import Path

LABELS = ["_silence", "_unknown", "backward", "spottieottie"]

_SCRIPT_DIR = Path(__file__).resolve().parent


def run_tflite_inference(interpreter, x):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    input_dtype = input_details[0]['dtype']
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    if input_dtype == np.int8:
        x_in = x / input_scale + input_zero_point
        x_in = np.clip(np.round(x_in), -128, 127).astype(np.int8)
    else:
        x_in = x.astype(np.float32)

    interpreter.set_tensor(input_index, x_in)
    interpreter.invoke()
    y = interpreter.get_tensor(output_index)

    if output_details[0]['dtype'] == np.int8:
        y = (y.astype(np.float32) - output_zero_point) * output_scale

    return y

def evaluate_tflite_model(tflite_path, x_data, y_true):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    preds = []
    for i in range(len(x_data)):
        x = x_data[i:i+1]
        y = run_tflite_inference(interpreter, x)
        preds.append(np.argmax(y, axis=1)[0])

    preds = np.array(preds)
    acc = np.mean(preds == y_true)
    return acc, preds


if __name__ == "__main__":
    _float_path = _SCRIPT_DIR / "kws_model_float.tflite"
    _dynamic_path = _SCRIPT_DIR / "kws_model_dynamic.tflite"
    _int8_path = _SCRIPT_DIR / "kws_model_int8.tflite"

    print("\n=== Interpreter input/output details ===")
    print("FLOAT model:")
    interp_float = tf.lite.Interpreter(model_path=str(_float_path))
    interp_float.allocate_tensors()
    print(interp_float.get_input_details())
    print(interp_float.get_output_details())

    print("\nDYNAMIC model (weights quantized, float I/O):")
    if _dynamic_path.is_file():
        interp_dyn = tf.lite.Interpreter(model_path=str(_dynamic_path))
        interp_dyn.allocate_tensors()
        print(interp_dyn.get_input_details())
        print(interp_dyn.get_output_details())
    else:
        print(f"  (missing {_dynamic_path.name}; run convert_to_tflite.py)")

    print("\nINT8 model:")
    interp_int8 = tf.lite.Interpreter(model_path=str(_int8_path))
    interp_int8.allocate_tensors()
    _in8 = interp_int8.get_input_details()
    _out8 = interp_int8.get_output_details()
    print(_in8)
    print(_out8)
    if _in8:
        _q = _in8[0].get("quantization", (0.0, 0))
        _scale, _zp = _q[0], _q[1]
        print(
            f"INT8 input dtype: {_in8[0]['dtype']} | "
            f"quantization (scale, zero_point): ({_scale}, {_zp})"
        )
        if _in8[0]["dtype"] != np.int8:
            print("  Warning: expected int8 input for full INT8 model.")
        if _scale == 0 or (isinstance(_scale, float) and abs(_scale) < 1e-10):
            print("  Warning: input scale is ~0; calibration may be wrong.")

    print("\n=== TFLite evaluation ===")

    _audio_path = _SCRIPT_DIR / "test_audio.npy"
    _labels_path = _SCRIPT_DIR / "y_true.npy"
    if not _audio_path.is_file() or not _labels_path.is_file():
        _cwd_audio = Path.cwd() / "test_audio.npy"
        _cwd_labels = Path.cwd() / "y_true.npy"
        if _cwd_audio.is_file() and _cwd_labels.is_file():
            _audio_path, _labels_path = _cwd_audio, _cwd_labels
        else:
            raise SystemExit(
                f"Need test_audio.npy and y_true.npy in {_SCRIPT_DIR} (saved automatically "
                "when speech_training.py finishes), or in the current working directory.\n"
                "Run: python speech_training.py from repo root, or cd training && python speech_training.py"
            )
    test_audio = np.load(_audio_path)
    y_true = np.load(_labels_path)

    float_tflite_acc, float_tflite_preds = evaluate_tflite_model(
        str(_float_path),
        test_audio.astype(np.float32),
        y_true,
    )
    print(f"Float TFLite accuracy: {float_tflite_acc:.4f}")

    if _dynamic_path.is_file():
        dynamic_tflite_acc, dynamic_tflite_preds = evaluate_tflite_model(
            str(_dynamic_path),
            test_audio.astype(np.float32),
            y_true,
        )
        print(f"Dynamic-range TFLite accuracy: {dynamic_tflite_acc:.4f}")
    else:
        dynamic_tflite_preds = None
        print("Dynamic-range TFLite: (skipped — no kws_model_dynamic.tflite)")

    int8_tflite_acc, int8_tflite_preds = evaluate_tflite_model(
        str(_int8_path),
        test_audio.astype(np.float32),
        y_true,
    )
    print(f"INT8 TFLite accuracy: {int8_tflite_acc:.4f}")
    print(
        "Float predicted counts:",
        {LABELS[i]: int(np.sum(float_tflite_preds == i)) for i in range(len(LABELS))},
    )
    if dynamic_tflite_preds is not None:
        print(
            "Dynamic predicted counts:",
            {LABELS[i]: int(np.sum(dynamic_tflite_preds == i)) for i in range(len(LABELS))},
        )
    print(
        "INT8 predicted counts:",
        {LABELS[i]: int(np.sum(int8_tflite_preds == i)) for i in range(len(LABELS))},
    )