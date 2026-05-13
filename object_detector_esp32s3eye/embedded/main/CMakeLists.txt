idf_component_register(
    SRCS "app_main_cifar.cc"
    INCLUDE_DIRS "."
    REQUIRES esp32-camera
    PRIV_REQUIRES esp_timer
    PRIV_REQUIRES esp_timer esp-tflite-micro
)
