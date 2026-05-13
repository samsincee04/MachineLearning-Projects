/* Day 4: I2S capture (C-style, known-good) + TFLite Micro INT8 KWS.
 *
 * This file is main.cpp (not main.c) because TensorFlow Lite Micro is a C++ API.
 * The ESP-IDF entry point app_main is declared with extern "C" below.
 *
 * Log-mel features: TFLM microfrontend (C) from tensorflow/tflite-micro
 * `experimental/microfrontend/lib` — see `tflite_upstream/` in this component and
 * `CMakeLists.txt` (the managed esp-tflite-micro component omits that subtree). */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"

/* TFLite Micro (C++) */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"

extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
}

/* -------------------------------------------------------------------------- */
/* Working audio capture (known-good: do not change without retest)           */
/* -------------------------------------------------------------------------- */

#define SAMPLE_RATE     16000
#define DMA_BUF_COUNT   3
#define DMA_BUF_LEN     300
#define READ_BUF_SIZE   1024   // 32-bit samples

#define I2S_BCK_PIN     GPIO_NUM_41
#define I2S_WS_PIN      GPIO_NUM_42
#define I2S_DATA_IN_PIN GPIO_NUM_2

/* -------------------------------------------------------------------------- */
/* Training-aligned assumptions (16 kHz audio; microfrontend / spectrogram)   */
/* -------------------------------------------------------------------------- */

#define MODEL_AUDIO_WINDOW_SAMPLES 16000
#define MODEL_WINDOW_SIZE_MS        64
#define MODEL_WINDOW_STEP_MS        48
#define MODEL_NUM_FILTERS            32

/* Feature tensor layout: [1, 20, 32, 1] INT8 — matches TFLite model input. */
#define kFeatureSliceSize        32
#define kFeatureSliceCount        20
#define kFeatureSliceStrideMs    48
#define kFeatureSliceDurationMs    64
#define kFeatureElementCount  (kFeatureSliceCount * kFeatureSliceSize) /* 640 */
#define kAudioSampleFrequency 16000

/* 64 ms analysis window = 1024 samples at 16 kHz; 48 ms stride = 768 samples. */
#define kFeatureSliceInputSamples 1024
/* Consecutive 64 ms analysis windows in the 1 s buffer overlap by 256 samples (768 stride).
 * Used only for building [1,20,32,1]; do not confuse with INFERENCE_HOP_SAMPLES. */
#define kFeatureBufferStrideSamples 768

/* Frontend state: 1 = reset before every 1024-sample slice (independent windows; may break
   temporal noise/PCAN). 0 = one FrontendReset() per full 1 s pass, then 20 ProcessSamples
   (tests whether per-slice reset caused silence misclassification / odd bias). */
#define RESET_FRONTEND_EACH_SLICE 0

/* How many full feature-generation passes (after success) get extended diagnostics. */
#define S_FEATURE_DEBUG_PASS_LIMIT 5

/* How much new audio before running another full 1 s feature + TFLM pass (not slice stride). */
#define INFERENCE_HOP_SAMPLES        1536

#define NUM_LABELS 4

/* Label indices (must match training / s_kws_labels). */
#define KWS_LABEL_SILENCE     0
#define KWS_LABEL_UNKNOWN     1
#define KWS_LABEL_BACKWARD    2
#define KWS_LABEL_SPOTTIEOTTIE 3

/* -------------------------------------------------------------------------- */
/* Live target declaration — tune only this block for deployment behavior.   */
/* -------------------------------------------------------------------------- */
/* Consecutive qualifying frames before declaring a target (keep at 2). */
#define TARGET_CONFIRM_FRAMES            2

/* Inferences to skip after a confirmed target (simple cooldown; raise to reduce repeats). */
#define TARGET_COOLDOWN_INFERENCES       6

/* Per-target minimum softmax score when raw argmax is that label (conservative defaults). */
#define TARGET_BACKWARD_SCORE_THRESHOLD       3.3f  /* "backward" */
#define TARGET_SPOTTIEOTTIE_SCORE_THRESHOLD   3.0f  /* "spottieottie" */

/* When raw top is backward or spottieottie: require target_score > unknown + this margin. */
#define TARGET_VS_UNKNOWN_MARGIN             0.75f

/* Tensor arena for TFLM (tune if AllocateTensors fails). */
#define TFLM_TENSOR_ARENA_SIZE (64 * 1024)

static int8_t s_feature_int8[kFeatureElementCount];

/* INT8 scaling: tflite micro_speech / micro_features convention */
static const int kFeatureValueScale  = 256;
static const int kFeatureValueDiv    = 666; /* round(25.6f * 26.0f) */

static FrontendState s_micro_features_state;
static bool          s_micro_features_initialized = false;
static int           s_feature_debug_hops         = 0; /* < S_FEATURE_DEBUG_PASS_LIMIT */

static const char *const s_kws_labels[NUM_LABELS] = {
    "_silence",
    "_unknown",
    "backward",
    "spottieottie",
};

static int16_t audio_ring[MODEL_AUDIO_WINDOW_SAMPLES];
static int     s_ring_count; /* 0 .. MODEL_AUDIO_WINDOW_SAMPLES */

static int new_samples_since_last_inference;

/* Live target declaration state (2-frame confirm + cooldown; not applied to _silence/_unknown). */
static int s_pending_target_label     = -1; /* KWS_LABEL_BACKWARD, KWS_LABEL_SPOTTIEOTTIE, or -1 */
static int s_pending_target_count     = 0;
static int s_target_cooldown_remaining = 0;
static int s_cooldown_last_label      = KWS_LABEL_BACKWARD;

/* Embedded model (symbol definitions in training/model_data.cc) */
extern unsigned char kws_model_int8_tflite[];
extern unsigned int  kws_model_int8_tflite_len;

/* TFLM state */
static const tflite::Model *s_tflm_model = nullptr;
static tflite::MicroInterpreter *s_tflm_interpreter = nullptr;
static TfLiteTensor *s_tflm_input = nullptr;
static TfLiteTensor *s_tflm_output = nullptr;
static bool s_tflm_init_ok = false;
static bool s_tflm_init_tried = false;

alignas(16) static uint8_t s_tflm_arena[TFLM_TENSOR_ARENA_SIZE];

static const char *TAG = "day4_stream";

/* --- Rolling window: keep most recent 1s at 16 kHz (oldest samples dropped) --- */
static void append_audio_chunk(const int16_t *chunk, int chunk_samples)
{
    if (chunk == NULL || chunk_samples <= 0) {
        return;
    }
    if (chunk_samples >= MODEL_AUDIO_WINDOW_SAMPLES) {
        const int16_t *tail = chunk + (chunk_samples - MODEL_AUDIO_WINDOW_SAMPLES);
        (void)memcpy(
            audio_ring,
            tail,
            (size_t)MODEL_AUDIO_WINDOW_SAMPLES * sizeof(int16_t)
        );
        s_ring_count = MODEL_AUDIO_WINDOW_SAMPLES;
        return;
    }
    {
        const int total = s_ring_count + chunk_samples;
        if (total <= MODEL_AUDIO_WINDOW_SAMPLES) {
            (void)memcpy(
                audio_ring + s_ring_count,
                chunk,
                (size_t)chunk_samples * sizeof(int16_t)
            );
            s_ring_count = total;
            return;
        }
        const int drop = total - MODEL_AUDIO_WINDOW_SAMPLES;
        if (drop < s_ring_count) {
            (void)memmove(
                audio_ring,
                audio_ring + drop,
                (size_t)(s_ring_count - drop) * sizeof(int16_t)
            );
            s_ring_count -= drop;
            (void)memcpy(
                audio_ring + s_ring_count,
                chunk,
                (size_t)chunk_samples * sizeof(int16_t)
            );
            s_ring_count += chunk_samples;
        } else {
            const int from_chunk = chunk_samples - MODEL_AUDIO_WINDOW_SAMPLES;
            (void)memcpy(
                audio_ring,
                chunk + from_chunk,
                (size_t)MODEL_AUDIO_WINDOW_SAMPLES * sizeof(int16_t)
            );
            s_ring_count = MODEL_AUDIO_WINDOW_SAMPLES;
        }
    }
}

static bool init_microfrontend(void)
{
    if (s_micro_features_initialized) {
        return true;
    }
    struct FrontendConfig config;
    FrontendFillConfigWithDefaults(&config);

    config.window.size_ms      = (size_t)kFeatureSliceDurationMs;
    config.window.step_size_ms  = (size_t)kFeatureSliceStrideMs;
    config.filterbank.num_channels     = 32;
    config.filterbank.lower_band_limit = 125.0f;
    config.filterbank.upper_band_limit = 7500.0f;

    config.noise_reduction.smoothing_bits        = 10;
    config.noise_reduction.even_smoothing      = 0.025f;
    config.noise_reduction.odd_smoothing        = 0.06f;
    config.noise_reduction.min_signal_remaining = 0.05f;

    config.pcan_gain_control.enable_pcan = 1;
    config.pcan_gain_control.strength   = 0.95f;
    config.pcan_gain_control.offset     = 80.0f;
    config.pcan_gain_control.gain_bits  = 21;

    config.log_scale.enable_log   = 1;
    config.log_scale.scale_shift  = 6;

    if (!FrontendPopulateState(&config, &s_micro_features_state, kAudioSampleFrequency)) {
        printf("Microfrontend init FAILED (FrontendPopulateState)\n");
        return false;
    }
    printf("Microfrontend init OK (window=%dms step=%dms, 32 Mel bins)\n",
           kFeatureSliceDurationMs, kFeatureSliceStrideMs);
    s_micro_features_initialized = true;
    return true;
}

/* Diagnostic: INT8 feature tensor stats (saturation, bias, spread vs training expectations). */
static void print_feature_stats(const int8_t *features, int count, const char *tag)
{
    if (features == NULL || count <= 0) {
        printf("[%s] (no features)\n", tag != NULL ? tag : "?");
        return;
    }
    int8_t  mn = features[0];
    int8_t  mx = features[0];
    int64_t sum = 0;
    int64_t abs_sum = 0;
    for (int i = 0; i < count; i++) {
        if (features[i] < mn) {
            mn = features[i];
        }
        if (features[i] > mx) {
            mx = features[i];
        }
        sum += features[i];
        abs_sum += (int64_t)llabs((long long)features[i]);
    }
    const double mean = (double)sum / (double)count;
    const double avg_abs = (double)abs_sum / (double)count;
    printf("[feat %s] n=%d min=%d max=%d mean=%.2f avg_abs=%.2f  first16:",
           tag != NULL ? tag : "?", count, (int)mn, (int)mx, mean, avg_abs);
    {
        int n = count < 16 ? count : 16;
        for (int i = 0; i < n; i++) {
            printf(" %d", (int)features[i]);
        }
    }
    printf("\n");
}

/* Shorter than print_i16_stats: rolling-window stats without flooding the log. */
static void print_audio_stats_short(const int16_t *samples, int count, const char *tag)
{
    if (samples == NULL || count <= 0) {
        printf("[%s] (no audio)\n", tag != NULL ? tag : "?");
        return;
    }
    int16_t min_val = samples[0];
    int16_t max_val = samples[0];
    int64_t abs_sum = 0;
    for (int i = 0; i < count; i++) {
        if (samples[i] < min_val) {
            min_val = samples[i];
        }
        if (samples[i] > max_val) {
            max_val = samples[i];
        }
        abs_sum += (int64_t)llabs((long long)samples[i]);
    }
    printf(
        "[audio %s] n=%d min=%d max=%d avg_abs=%lld  first8:",
        tag != NULL ? tag : "?", count, min_val, max_val, abs_sum / (int64_t)count
    );
    {
        int n = count < 8 ? count : 8;
        for (int i = 0; i < n; i++) {
            printf(" %d", (int)samples[i]);
        }
    }
    printf("\n");
}

/* Grep/capture-friendly: whole-tensor and per-slice stats (first S_FEATURE_DEBUG_PASS_LIMIT
 * successful passes). clamp_* = bins that would clip before INT8 (same as embedded quant path). */
static void compute_int8_tensor_stats(
    const int8_t *d, int n, int8_t *out_min, int8_t *out_max, double *out_mean, double *out_avg_abs
)
{
    if (d == NULL || n <= 0 || out_min == NULL || out_max == NULL || out_mean == NULL
        || out_avg_abs == NULL) {
        return;
    }
    int8_t  mn   = d[0];
    int8_t  mx   = d[0];
    int64_t sum  = 0;
    int64_t aabs = 0;
    for (int i = 0; i < n; i++) {
        if (d[i] < mn) {
            mn = d[i];
        }
        if (d[i] > mx) {
            mx = d[i];
        }
        sum += d[i];
        aabs += (int64_t)llabs((long long)d[i]);
    }
    *out_min   = mn;
    *out_max   = mx;
    *out_mean  = (double)sum / (double)n;
    *out_avg_abs = (double)aabs / (double)n;
}

static void compute_int8_row_stats(
    const int8_t *row, int n, int8_t *rmin, int8_t *rmax, double *rmean, double *ravg_abs
)
{
    if (row == NULL || n <= 0) {
        return;
    }
    compute_int8_tensor_stats(row, n, rmin, rmax, rmean, ravg_abs);
}

static void print_feature_dump_machine_readable(
    int pass,
    const int8_t *feat_640,
    int clamp_lo_640,
    int clamp_hi_640
)
{
    if (feat_640 == NULL) {
        return;
    }
    int8_t  mn, mx;
    double  mean, avg_abs;
    compute_int8_tensor_stats(
        feat_640, kFeatureElementCount, &mn, &mx, &mean, &avg_abs
    );
    printf(
        "FEATURE_SUMMARY pass=%d min=%d max=%d mean=%.4f avg_abs=%.4f clamp_lo=%d "
        "clamp_hi=%d\n",
        pass, (int)mn, (int)mx, mean, avg_abs, clamp_lo_640, clamp_hi_640
    );
    printf("FEATURE_HEAD pass=%d values=", pass);
    for (int i = 0; i < 32; i++) {
        printf("%d%s", (int)feat_640[i], (i < 31) ? "," : "");
    }
    printf("\n");
    for (int s = 0; s < kFeatureSliceCount; s++) {
        const int8_t *row = feat_640 + s * kFeatureSliceSize;
        int8_t        smn, smx;
        double        smean, savg;
        compute_int8_row_stats(row, kFeatureSliceSize, &smn, &smx, &smean, &savg);
        printf(
            "FEATURE_SLICE pass=%d slice=%d min=%d max=%d mean=%.4f avg_abs=%.4f\n",
            pass, s, (int)smn, (int)smx, smean, savg
        );
    }
}

/* One 64 ms / 1024-sample slice -> one row of 32 uint16 -> 32 x INT8. */
/* slice_index: for logging only when print_slice_diagnostics is true (first 2–3 slices, first
 * S_FEATURE_DEBUG_PASS_LIMIT passes). acc_clamp_*: optional 640-tensor quant clamp totals. */
static bool generate_one_feature_slice(
    const int16_t *input, int input_size, int8_t *output_32, int slice_index,
    bool print_slice_diagnostics, int *acc_clamp_lo, int *acc_clamp_hi
)
{
    if (!s_micro_features_initialized) {
        return false;
    }
    if (input == NULL || output_32 == NULL || input_size < kFeatureSliceInputSamples) {
        return false;
    }

#if RESET_FRONTEND_EACH_SLICE
    /* Compare against pass-level reset: each slice re-inits the frontend. */
    FrontendReset(&s_micro_features_state);
#endif

    size_t         num_read = 0;
    FrontendOutput out =
        FrontendProcessSamples(&s_micro_features_state, input,
                              (size_t)kFeatureSliceInputSamples, &num_read);
    (void)num_read; /* expect window to consume 1024 after reset */
    if (out.values == NULL || out.size < (size_t)kFeatureSliceSize) {
        printf("Microfrontend: output too small (size=%u, need>=%d)\n",
               (unsigned)out.size, kFeatureSliceSize);
        return false;
    }
    int clamp_low = 0;
    int clamp_high = 0;
    int8_t smin = 127;
    int8_t smax = (int8_t) -128;
    for (int i = 0; i < kFeatureSliceSize; i++) {
        int32_t raw = ((int32_t)out.values[i] * kFeatureValueScale + (kFeatureValueDiv / 2))
                    / kFeatureValueDiv;
        int32_t v = raw - 128;
        if (v < -128) {
            clamp_low++;
            v = -128;
        } else if (v > 127) {
            clamp_high++;
            v = 127;
        }
        output_32[i] = (int8_t)v;
        if (output_32[i] < smin) {
            smin = output_32[i];
        }
        if (output_32[i] > smax) {
            smax = output_32[i];
        }
    }
    if (acc_clamp_lo != NULL) {
        *acc_clamp_lo += clamp_low;
    }
    if (acc_clamp_hi != NULL) {
        *acc_clamp_hi += clamp_high;
    }
    if (print_slice_diagnostics) {
        printf(
            "  [slice %d] int8 min=%d max=%d  clamp<=%d  clamp>=%d\n",
            slice_index, (int)smin, (int)smax, clamp_low, clamp_high
        );
    }
    return true;
}

/* 20 time slices (48 ms hop) x 32 bins = 640 INT8 in row-major (slice-major) layout. */
static bool generate_features_from_audio(const int16_t *audio_1s, int audio_samples)
{
    if (audio_1s == NULL) {
        return false;
    }
    if (audio_samples < MODEL_AUDIO_WINDOW_SAMPLES) {
        return false;
    }
    if (!init_microfrontend()) {
        return false;
    }

    const bool do_pass_diag =
        (s_feature_debug_hops < S_FEATURE_DEBUG_PASS_LIMIT);
    if (do_pass_diag) {
        print_audio_stats_short(
            audio_1s, MODEL_AUDIO_WINDOW_SAMPLES, "roll1s (pre-frontend)"
        );
    }

/* With RESET_FRONTEND_EACH_SLICE=0, one clean state per 1 s window (matches 20 strided
   1024-sample extractions; flip macro to 1 to compare "fresh frontend every slice"). */
#if !RESET_FRONTEND_EACH_SLICE
    FrontendReset(&s_micro_features_state);
#endif

    int acc_clamp_lo_640 = 0;
    int acc_clamp_hi_640 = 0;
    for (int s = 0; s < kFeatureSliceCount; s++) {
        const int    slice_start = s * kFeatureBufferStrideSamples;
        const int16_t *slice_in  = audio_1s + slice_start;
        int8_t        *slice_out = s_feature_int8 + s * kFeatureSliceSize;
        if (slice_start + kFeatureSliceInputSamples > audio_samples) {
            printf("generate_features: slice %d OOB (start %d, len %d)\n", s, slice_start,
                   kFeatureSliceInputSamples);
            return false;
        }
        const bool slice_clamp_log = do_pass_diag && (s < 3);
        if (!generate_one_feature_slice(
                slice_in, kFeatureSliceInputSamples, slice_out, s, slice_clamp_log,
                do_pass_diag ? &acc_clamp_lo_640 : NULL, do_pass_diag ? &acc_clamp_hi_640 : NULL
            )) {
            return false;
        }
    }

    if (do_pass_diag) {
        const int pass = s_feature_debug_hops;
        print_feature_dump_machine_readable(
            pass, s_feature_int8, acc_clamp_lo_640, acc_clamp_hi_640
        );
        print_feature_stats(
            s_feature_int8, kFeatureElementCount, "640 after 20 slices (live)"
        );
        s_feature_debug_hops++;
    }

    return true;
}

static void log_tensor_mismatch(const char *name, const TfLiteTensor *t, const char *expect)
{
    printf("TFLM %s mismatch (expected %s): type=%d dims=%d", name, expect, t->type, t->dims->size);
    for (int i = 0; i < t->dims->size; i++) {
        printf("%c%d", i ? ',' : '[', t->dims->data[i]);
    }
    printf("]\n");
}

static bool validate_input_tensor(const TfLiteTensor *in)
{
    if (in->type != kTfLiteInt8) {
        return false;
    }
    if (in->dims->size != 4) {
        return false;
    }
    if (in->dims->data[0] != 1 || in->dims->data[1] != 20
        || in->dims->data[2] != 32 || in->dims->data[3] != 1) {
        return false;
    }
    return true;
}

static bool validate_output_tensor(const TfLiteTensor *out)
{
    if (out->type != kTfLiteInt8) {
        return false;
    }
    if (out->dims->size != 2) {
        return false;
    }
    if (out->dims->data[0] != 1 || out->dims->data[1] != 4) {
        return false;
    }
    return true;
}

/* KWS model (Keras) uses Conv2D + MaxPool2D + Dense, not micro_speech-style depthwise-only. */
static bool init_tflm_inference(void)
{
    if (s_tflm_init_tried) {
        return s_tflm_init_ok;
    }
    s_tflm_init_tried = true;

    s_tflm_model = tflite::GetModel(kws_model_int8_tflite);
    if (s_tflm_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("TFLM init FAIL: model schema %lu != TFLITE_SCHEMA_VERSION %d\n",
               (unsigned long)s_tflm_model->version(), TFLITE_SCHEMA_VERSION);
        s_tflm_init_ok = false;
        return false;
    }
    (void)kws_model_int8_tflite_len; /* model length; GetModel does not need it */

    /* Conv2D/MaxPool2D for KWS; Add/Mul may be used in quantized graphs. */
    static tflite::MicroMutableOpResolver<9> s_resolver;
    if (s_resolver.AddConv2D() != kTfLiteOk) {
        printf("TFLM init FAIL: AddConv2D\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (s_resolver.AddMaxPool2D() != kTfLiteOk) {
        printf("TFLM init FAIL: AddMaxPool2D\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (s_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
        printf("TFLM init FAIL: AddDepthwiseConv2D\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (s_resolver.AddFullyConnected() != kTfLiteOk) {
        printf("TFLM init FAIL: AddFullyConnected\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (s_resolver.AddReshape() != kTfLiteOk) {
        printf("TFLM init FAIL: AddReshape\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (s_resolver.AddSoftmax() != kTfLiteOk) {
        printf("TFLM init FAIL: AddSoftmax\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (s_resolver.AddMul() != kTfLiteOk) {
        printf("TFLM init FAIL: AddMul\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (s_resolver.AddAdd() != kTfLiteOk) {
        printf("TFLM init FAIL: AddAdd\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (s_resolver.AddMean() != kTfLiteOk) {
        printf("TFLM init FAIL: AddMean\n");
        s_tflm_init_ok = false;
        return false;
    }

    static tflite::MicroInterpreter s_interpreter(
        s_tflm_model, s_resolver, s_tflm_arena, sizeof(s_tflm_arena)
    );
    s_tflm_interpreter = &s_interpreter;

    if (s_tflm_interpreter->AllocateTensors() != kTfLiteOk) {
        printf("TFLM init FAIL: AllocateTensors (try larger TFLM_TENSOR_ARENA_SIZE)\n");
        s_tflm_init_ok = false;
        return false;
    }

    s_tflm_input  = s_tflm_interpreter->input(0);
    s_tflm_output = s_tflm_interpreter->output(0);
    if (s_tflm_input == NULL || s_tflm_output == NULL) {
        printf("TFLM init FAIL: null input/output tensor\n");
        s_tflm_init_ok = false;
        return false;
    }
    if (!validate_input_tensor(s_tflm_input)) {
        log_tensor_mismatch("input", s_tflm_input, "INT8 [1,20,32,1]");
        s_tflm_init_ok = false;
        return false;
    }
    if (!validate_output_tensor(s_tflm_output)) {
        log_tensor_mismatch("output", s_tflm_output, "INT8 [1,4]");
        s_tflm_init_ok = false;
        return false;
    }

    printf("TFLM init OK\n");
    printf("input shape OK: [1,20,32,1] dtype INT8\n");
    printf("output shape OK: [1,4] dtype INT8\n");
    s_tflm_init_ok = true;
    return true;
}

static bool run_inference_and_get_scores(float *scores_out, int score_count)
{
    if (scores_out == NULL || score_count < NUM_LABELS) {
        return false;
    }
    if (!init_tflm_inference() || !s_tflm_init_ok || s_tflm_input == NULL || s_tflm_output == NULL) {
        return false;
    }

    int8_t *in_ptr = tflite::GetTensorData<int8_t>(s_tflm_input);
    if (in_ptr == NULL) {
        return false;
    }
    (void)memcpy(in_ptr, s_feature_int8, (size_t)kFeatureElementCount * sizeof(int8_t));

    if (s_tflm_interpreter->Invoke() != kTfLiteOk) {
        printf("TFLM Invoke failed\n");
        return false;
    }

    int8_t *out_ptr = tflite::GetTensorData<int8_t>(s_tflm_output);
    if (out_ptr == NULL) {
        return false;
    }
    const float oscale = s_tflm_output->params.scale;
    const int   ozp   = s_tflm_output->params.zero_point;
    for (int i = 0; i < NUM_LABELS; i++) {
        scores_out[i] = (static_cast<float>(out_ptr[i]) - static_cast<float>(ozp)) * oscale;
    }
    return true;
}

static int argmax_f(const float *x, int n)
{
    if (x == NULL || n <= 0) {
        return -1;
    }
    int imax = 0;
    for (int i = 1; i < n; i++) {
        if (x[i] > x[imax]) {
            imax = i;
        }
    }
    return imax;
}

static void log_prediction(const float *scores, int n_scores)
{
    if (scores == NULL) {
        printf("prediction: (invalid)\n");
        return;
    }
    int k = n_scores;
    if (k > NUM_LABELS) {
        k = NUM_LABELS;
    }
    if (k <= 0) {
        printf("prediction: (invalid)\n");
        return;
    }
    const int top = argmax_f(scores, k);
    printf("scores: ");
    for (int i = 0; i < k; i++) {
        printf("%s%.4f", i ? " " : "", (double)scores[i]);
    }
    if (top >= 0 && top < NUM_LABELS) {
        printf(" | top: %s (%.4f)\n", s_kws_labels[top], (double)scores[top]);
    } else {
        printf(" | top: (n/a)\n");
    }
}

static void log_raw_declared_label(const float *scores, int n)
{
    if (scores == NULL) {
        return;
    }
    int k = n;
    if (k > NUM_LABELS) {
        k = NUM_LABELS;
    }
    if (k <= 0) {
        return;
    }
    const int top = argmax_f(scores, k);
    if (top >= 0 && top < NUM_LABELS) {
        printf("decl_raw: %s (%.4f)\n", s_kws_labels[top], (double)scores[top]);
    } else {
        printf("decl_raw: (n/a)\n");
    }
}

/* Live policy helpers: label index is KWS_LABEL_* (see s_kws_labels). */
static bool kws_label_is_target(int label_idx)
{
    return label_idx == KWS_LABEL_BACKWARD || label_idx == KWS_LABEL_SPOTTIEOTTIE;
}

static float kws_target_score_threshold(int label_idx)
{
    if (label_idx == KWS_LABEL_BACKWARD) {
        return TARGET_BACKWARD_SCORE_THRESHOLD;
    }
    if (label_idx == KWS_LABEL_SPOTTIEOTTIE) {
        return TARGET_SPOTTIEOTTIE_SCORE_THRESHOLD;
    }
    return 0.0f;
}

/* Raw top is a target: require score > threshold and score > unknown + margin. */
static void live_reset_pending_target(void)
{
    s_pending_target_label   = -1;
    s_pending_target_count   = 0;
}

static void log_live_target_declaration(const float *scores, int n)
{
    if (scores == NULL) {
        printf("decl_live: (invalid)\n");
        return;
    }
    int k = n;
    if (k > NUM_LABELS) {
        k = NUM_LABELS;
    }
    if (k <= 0) {
        printf("decl_live: (invalid)\n");
        return;
    }

    const int raw_top = argmax_f(scores, k);
    if (raw_top < 0) {
        printf("decl_live: none (argmax n/a)\n");
        return;
    }

    /* Cooldown: same as before — counts down every inference; clears pending if raw target changes. */
    if (s_target_cooldown_remaining > 0) {
        printf(
            "decl_live: cooldown(%d) last=%s\n",
            s_target_cooldown_remaining,
            (s_cooldown_last_label >= 0 && s_cooldown_last_label < NUM_LABELS) ? s_kws_labels[s_cooldown_last_label] : "?"
        );
        s_target_cooldown_remaining--;
        if (s_pending_target_label >= 0 && raw_top != s_pending_target_label) {
            live_reset_pending_target();
        }
        return;
    }

    /* _silence / _unknown / non-target: no extra thresholds; just clear pending and exit. */
    if (!kws_label_is_target(raw_top)) {
        live_reset_pending_target();
        printf("decl_live: none raw=%s\n", s_kws_labels[raw_top]);
        return;
    }

    const float target_score  = scores[raw_top];
    const float unknown_score = scores[KWS_LABEL_UNKNOWN];
    const float th            = kws_target_score_threshold(raw_top);

    if (target_score <= th) {
        live_reset_pending_target();
        printf("decl_live: none reason=target_below_threshold\n");
        return;
    }
    if (target_score <= unknown_score + TARGET_VS_UNKNOWN_MARGIN) {
        live_reset_pending_target();
        printf("decl_live: none reason=target_not_above_unknown_margin\n");
        return;
    }

    /* 2-frame confirmation (unchanged structure). */
    if (raw_top == s_pending_target_label) {
        s_pending_target_count++;
    } else {
        s_pending_target_label = raw_top;
        s_pending_target_count   = 1;
    }

    if (s_pending_target_count >= TARGET_CONFIRM_FRAMES) {
        printf(
            "decl_live: %s confirmed %d/%d\n",
            s_kws_labels[raw_top],
            TARGET_CONFIRM_FRAMES,
            TARGET_CONFIRM_FRAMES
        );
        s_cooldown_last_label       = raw_top;
        s_target_cooldown_remaining = TARGET_COOLDOWN_INFERENCES;
        live_reset_pending_target();
        return;
    }

    printf(
        "decl_live: none pending=%s count=%d\n",
        s_kws_labels[raw_top],
        s_pending_target_count
    );
}

/* -------------------------------------------------------------------------- */
/* Debug helpers (known-good capture visibility)                              */
/* -------------------------------------------------------------------------- */

static void print_i32_stats(const int32_t *samples, int sample_count, const char *tag)
{
    if (sample_count <= 0) {
        printf("[%s] no samples\n", tag);
        return;
    }

    int32_t min_val = samples[0];
    int32_t max_val = samples[0];
    long long abs_sum = 0;

    for (int i = 0; i < sample_count; i++) {
        if (samples[i] < min_val) {
            min_val = samples[i];
        }
        if (samples[i] > max_val) {
            max_val = samples[i];
        }
        abs_sum += llabs((long long)samples[i]);
    }

    printf("\n[%s] count=%d min=%ld max=%ld avg_abs=%lld\n",
           tag, sample_count, (long)min_val, (long)max_val, abs_sum / sample_count);

    printf("[%s] first 10: ", tag);
    int limit = sample_count < 10 ? sample_count : 10;
    for (int i = 0; i < limit; i++) {
        printf("%ld ", (long)samples[i]);
    }
    printf("\n");
}

static void print_i16_stats(const int16_t *samples, int sample_count, const char *tag)
{
    if (sample_count <= 0) {
        printf("[%s] no samples\n", tag);
        return;
    }

    int16_t min_val = samples[0];
    int16_t max_val = samples[0];
    long long abs_sum = 0;

    for (int i = 0; i < sample_count; i++) {
        if (samples[i] < min_val) {
            min_val = samples[i];
        }
        if (samples[i] > max_val) {
            max_val = samples[i];
        }
        abs_sum += llabs((long long)samples[i]);
    }

    printf("[%s] count=%d min=%d max=%d avg_abs=%lld\n",
           tag, sample_count, min_val, max_val, abs_sum / sample_count);

    printf("[%s] first 10: ", tag);
    int limit = sample_count < 10 ? sample_count : 10;
    for (int i = 0; i < limit; i++) {
        printf("%d ", samples[i]);
    }
    printf("\n");
}

static void audio_task(void *arg)
{
    (void)arg;

    i2s_chan_handle_t rx_handle;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num  = DMA_BUF_COUNT;
    chan_cfg.dma_frame_num = DMA_BUF_LEN;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    /* Standard I2S, 32-bit slot, stereo framing, left channel only. */
    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_STEREO
        ),
        .gpio_cfg = {
            .mclk         = I2S_GPIO_UNUSED,
            .bclk         = I2S_BCK_PIN,
            .ws           = I2S_WS_PIN,
            .dout         = I2S_GPIO_UNUSED,
            .din          = I2S_DATA_IN_PIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));

    static int32_t raw[READ_BUF_SIZE];
    static int16_t pcm[READ_BUF_SIZE];
    static float   scores[NUM_LABELS];

    size_t bytes_read  = 0;
    int    debug_blocks = 0;

    while (1) {
        esp_err_t ret = i2s_channel_read(
            rx_handle,
            raw,
            sizeof(raw),
            &bytes_read,
            pdMS_TO_TICKS(1000)
        );

        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "i2s_channel_read failed: %s", esp_err_to_name(ret));
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        if (bytes_read == 0) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }

        int samples = (int)(bytes_read / sizeof(int32_t));
        if (samples <= 0) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        if (samples > READ_BUF_SIZE) {
            samples = READ_BUF_SIZE;
        }

        for (int i = 0; i < samples; ++i) {
            pcm[i] = (int16_t)(raw[i] >> 14);
        }

        /* Rolling 1s buffer, then hop counter (match prior known-good order). */
        append_audio_chunk(pcm, samples);
        new_samples_since_last_inference += samples;
        const bool buffer_warm   = (s_ring_count == MODEL_AUDIO_WINDOW_SAMPLES);
        const bool hop_ready     = buffer_warm
            && (new_samples_since_last_inference >= INFERENCE_HOP_SAMPLES);
        bool       did_infer     = false;

        if (hop_ready) {
            (void)generate_features_from_audio(audio_ring, MODEL_AUDIO_WINDOW_SAMPLES);
            if (run_inference_and_get_scores(scores, NUM_LABELS)) {
                did_infer = true;
            }
            new_samples_since_last_inference = 0;
        }

        if (debug_blocks < 8) {
            printf("bytes_read=%u samples=%d\n", (unsigned)bytes_read, samples);
            print_i32_stats(raw, samples, "raw_i32");
            printf("[raw_i32 hex] first 10: ");
            int limit = samples < 10 ? samples : 10;
            for (int i = 0; i < limit; i++) {
                printf("0x%08lx ", (unsigned long)raw[i]);
            }
            printf("\n");
            print_i16_stats(pcm, samples, "pcm_i16_shift14");
            debug_blocks++;
            vTaskDelay(pdMS_TO_TICKS(1000));
        } else {
            const char *infer_s = (hop_ready && did_infer) ? "yes" : "no";
            printf("stream step: samples=%d infer=%s\n", samples, infer_s);
            if (did_infer) {
                log_prediction(scores, NUM_LABELS);
                log_raw_declared_label(scores, NUM_LABELS);
                log_live_target_declaration(scores, NUM_LABELS);
            }
            /* Slightly more cooperative delay than 1 ms to ease task watchdog under continuous
             * inference; tune once silence/false-trigger root cause is clear. */
            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }
}

extern "C" void app_main(void)
{
    /* Unpinned: scheduler can place audio_task on either core; avoids pinning to the second CPU. */
    (void)xTaskCreate(audio_task, "audio_task", 12288, NULL, 5, NULL);
}
