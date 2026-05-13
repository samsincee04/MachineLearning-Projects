# main.cpp: TFLite Micro (C++), model bytes, and vendored tflite-micro microfrontend
# (upstream experimental/microfrontend is not shipped inside espressif__esp-tflite-micro;
# tflite_upstream/ holds tensorflow/lite/experimental/microfrontend/lib/*.[cc])
set(_mf_lib "${CMAKE_CURRENT_LIST_DIR}/tflite_upstream/tensorflow/lite/experimental/microfrontend/lib")
file(GLOB _mf_c "${_mf_lib}/*.c")
file(GLOB _mf_cc "${_mf_lib}/*.cc")
list(FILTER _mf_c EXCLUDE REGEX ".*_io\\.c$|.*_main\\.c$|.*memmap.*|.*_test\\.c$")
list(FILTER _mf_cc EXCLUDE REGEX ".*_test\\.cc$")

idf_component_register(
    SRCS
        "main.cpp"
        "${CMAKE_SOURCE_DIR}/../training/model_data.cc"
        ${_mf_c}
        ${_mf_cc}
    INCLUDE_DIRS
        "."
        "${CMAKE_CURRENT_LIST_DIR}/tflite_upstream"
    REQUIRES
        esp_driver_i2s
        espressif__esp-tflite-micro
)
