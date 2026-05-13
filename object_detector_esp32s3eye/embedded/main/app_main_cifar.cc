#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_camera.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

static const char *TAG = "shoe_model";

constexpr int kModelWidth = 48;
constexpr int kModelHeight = 48;
constexpr int kInputLen = kModelWidth * kModelHeight * 3;

constexpr int kTensorArenaSize = 900 * 1024;
static uint8_t* tensor_arena = nullptr;

constexpr float INPUT_SCALE = 0.003921568859368563f;
constexpr int INPUT_ZERO_POINT = -128;

constexpr float OUTPUT_SCALE = 0.00390625f;
constexpr int OUTPUT_ZERO_POINT = -128;

// ESP32-S3-EYE camera pins
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD     4
#define CAM_PIN_SIOC     5
#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2       8
#define CAM_PIN_D1       9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC    6
#define CAM_PIN_HREF     7
#define CAM_PIN_PCLK    13

static esp_err_t init_camera() {
    camera_config_t config = {};

    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.sccb_i2c_port = 1;

    config.pin_d7 = CAM_PIN_D7;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d0 = CAM_PIN_D0;

    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_pclk = CAM_PIN_PCLK;

    config.xclk_freq_hz = 10000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;

    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QQVGA;   // 160x120
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    return esp_camera_init(&config);
}
static inline int8_t quantize_u8_to_int8(uint8_t v) {
    int q = (int)(v / INPUT_SCALE + INPUT_ZERO_POINT + 0.5f);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static void preprocess_rgb565_to_int8(const camera_fb_t* fb, int8_t* dst) {
    const int src_w = fb->width;   // expected 160
    const int src_h = fb->height;  // expected 120

    const uint8_t* buf = fb->buf;

    // center-crop a square from 160x120 -> 120x120, then downsample to 48x48
    const int crop_size = (src_h < src_w) ? src_h : src_w;   // 120
    const int x_off = (src_w - crop_size) / 2;               // 20
    const int y_off = (src_h - crop_size) / 2;               // 0

    for (int dy = 0; dy < kModelHeight; ++dy) {
        for (int dx = 0; dx < kModelWidth; ++dx) {
            const int sx = x_off + (dx * crop_size) / kModelWidth;
            const int sy = y_off + (dy * crop_size) / kModelHeight;

            const int pixel_index = (sy * src_w + sx) * 2;

            const uint8_t lo = buf[pixel_index];
            const uint8_t hi = buf[pixel_index + 1];
            const uint16_t pixel = ((uint16_t)hi << 8) | lo;

            uint8_t r5 = (pixel >> 11) & 0x1F;
            uint8_t g6 = (pixel >> 5) & 0x3F;
            uint8_t b5 = pixel & 0x1F;

            uint8_t r8 = (r5 * 255) / 31;
            uint8_t g8 = (g6 * 255) / 63;
            uint8_t b8 = (b5 * 255) / 31;

            const int out_idx = (dy * kModelWidth + dx) * 3;
            dst[out_idx + 0] = quantize_u8_to_int8(r8);
            dst[out_idx + 1] = quantize_u8_to_int8(g8);
            dst[out_idx + 2] = quantize_u8_to_int8(b8);
        }
    }
}

extern "C" void app_main(void) {
    esp_err_t cam_err = init_camera();
    if (cam_err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", cam_err);
        return;
    }

    const tflite::Model* tfl_model = tflite::GetModel(model_data);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch!");
        return;
    }

    static tflite::MicroMutableOpResolver<11> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddMean();

    tensor_arena = (uint8_t*) heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM");
        return;
    }

    static tflite::MicroInterpreter interpreter(
        tfl_model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed!");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    ESP_LOGI(TAG, "Input shape: [%d, %d, %d, %d]",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3]);

    ESP_LOGI(TAG, "Output shape: [%d, %d]",
             output->dims->data[0], output->dims->data[1]);

    ESP_LOGI(TAG, "Input type: %d, Output type: %d", input->type, output->type);
    ESP_LOGI(TAG, "Starting live camera inference loop...");

    int iteration = 0;

    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Frame capture failed");
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        preprocess_rgb565_to_int8(fb, input->data.int8);

        const int64_t t1 = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed!");
            esp_camera_fb_return(fb);
            break;
        }
        const int64_t t2 = esp_timer_get_time();
        const int64_t inference_us = t2 - t1;

        const int8_t raw_output = output->data.int8[0];
        const float score = (raw_output - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
        const int predicted = (score >= 0.5f) ? 1 : 0;
        const char* predicted_name = predicted ? "shoe" : "no_shoe";

        printf("\n--- Live Iteration %d ---\n", iteration + 1);
        printf("Frame size: %ux%u\n", fb->width, fb->height);
        printf("Raw output: %d\n", raw_output);
        printf("Output score: %.6f\n", score);
        printf("Predicted: %s (%d)\n", predicted_name, predicted);
        printf("Inference time: %lld us (%.2f ms)\n",
               inference_us, inference_us / 1000.0f);

        esp_camera_fb_return(fb);
        iteration++;
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
