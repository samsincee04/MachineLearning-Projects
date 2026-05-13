import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="project1_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
