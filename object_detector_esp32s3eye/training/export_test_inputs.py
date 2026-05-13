from pathlib import Path
import numpy as np
from tensorflow import keras

IMG_SIZE = (96, 96)

INPUT_SCALE = 0.003921568859368563
INPUT_ZERO_POINT = -128

shoe_path = sorted((Path("dataset/processed/test/shoe")).glob("*"))[0]
no_shoe_path = sorted((Path("dataset/processed/test/no_shoe")).glob("*"))[0]

def load_img_array(path):
    img = keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img).astype("float32") / 255.0
    arr_q = np.round(arr / INPUT_SCALE + INPUT_ZERO_POINT).astype(np.int8)
    return arr_q

shoe_img = load_img_array(shoe_path)
no_shoe_img = load_img_array(no_shoe_path)

header_path = Path("../embedded/main/test_inputs.h")

with open(header_path, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")

    f.write("// Input 1: shoe\n")
    f.write(f"const int8_t input_shoe[{shoe_img.size}] = {{\n")
    f.write(", ".join(str(int(x)) for x in shoe_img.flatten()))
    f.write("\n};\n\n")

    f.write("// Input 2: no_shoe\n")
    f.write(f"const int8_t input_no_shoe[{no_shoe_img.size}] = {{\n")
    f.write(", ".join(str(int(x)) for x in no_shoe_img.flatten()))
    f.write("\n};\n\n")

    f.write('const char* class_names[2] = {"no_shoe", "shoe"};\n')

print(f"Wrote test inputs to {header_path}")
print(f"shoe image: {shoe_path.name}")
print(f"no_shoe image: {no_shoe_path.name}")
