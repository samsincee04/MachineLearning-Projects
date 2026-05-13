from pathlib import Path

TFLITE_PATH = Path("project1_model_int8.tflite")
HEADER_PATH = Path("../embedded/main/model.h")

if not TFLITE_PATH.exists():
    raise FileNotFoundError(f"Missing TFLite model: {TFLITE_PATH}")

data = TFLITE_PATH.read_bytes()

with open(HEADER_PATH, "w") as f:
    f.write("#pragma once\n")
    f.write(f"const unsigned int model_len = {len(data)};\n")
    f.write("alignas(8) const unsigned char model_data[] = {\n")
    f.write(", ".join(str(b) for b in data))
    f.write("\n};\n")

print(f"Wrote header to {HEADER_PATH}")
