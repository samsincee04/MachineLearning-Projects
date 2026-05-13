# Streaming Keyword Spotter for "backward" and "spottieottie"

## Overview

This project implements a streaming keyword spotting system for embedded deployment. The final system detects two target words:

- `backward` - Speech Commands target word
- `spottieottie` - custom target word

The deployed classifier uses four classes:

- `_silence`
- `_unknown`
- `backward`
- `spottieottie`

The workflow included data collection, preprocessing, TensorFlow training, quantization to TFLite, export to C++ model data, and live deployment on embedded hardware using a streaming microphone pipeline.

---

## Repository Structure

```text
keyword-spotter/
├── training/       # training, preprocessing, augmentation, quantization, evaluation scripts
├── embedded/       # ESP-IDF embedded deployment project
├── final_model/    # final exported models, plots, and summary artifacts
├── README.md
└── .gitignore
```

## Final Model Setup

Target words:

- `backward`
- `spottieottie`

Final class list:

- `_silence`
- `_unknown`
- `backward`
- `spottieottie`

Audio/frontend configuration:

- Sampling rate: 16 kHz
- Clip length: 1 second
- Window size: 64 ms
- Window step: 48 ms
- Filterbank channels: 32
- Input tensor shape: `(20, 32, 1)`

## Final Software Results

- Training accuracy: 98.42%
- Validation accuracy: 94.85%
- Test accuracy: 87.63%

Per-class TensorFlow results:

- `_silence`: TPR 1.0000, FPR 0.1500
- `_unknown`: TPR 0.7073, FPR 0.0000
- `backward`: TPR 1.0000, FPR 0.0000
- `spottieottie`: TPR 1.0000, FPR 0.0000

## Quantized Models

The final exported model variants are stored in `final_model/`:

- `kws_model_float.tflite`
- `kws_model_dynamic.tflite`
- `kws_model_int8.tflite`

The INT8 model was used for embedded deployment.

## Embedded Streaming Behavior

The embedded system runs continuously on live microphone audio using:

- rolling 1-second audio buffer
- microfrontend feature extraction
- INT8 TFLite Micro inference
- lightweight live decision logic

Final live decision logic:

- 2-frame confirmation
- short cooldown after detection
- per-target thresholds
- target-vs-unknown margin rule

## Noise / False Alarm Testing

False alarm testing was run for 10 minutes per condition.

Quiet (~35 dB):

- backward false alarms: 0
- spottieottie false alarms: 1
- total: 6 FA/hour

Medium (~56 dB):

- backward false alarms: 6
- spottieottie false alarms: 8
- total: 84 FA/hour

High (~69 dB):

- backward false alarms: 2
- spottieottie false alarms: 2
- total: 24 FA/hour

## Training Scripts

Main useful scripts in `training/`:

- `speech_training.py`
- `augment_audio.py`
- `preprocess_audio.py`
- `record_audio.py`
- `convert_to_tflite.py`
- `evaluate_tflite.py`
- `compare_silence_features.py`

## Comparing On-Device Features vs Training

To compare Python-side and on-device frontend behavior:

```bash
.venv/bin/python training/compare_silence_features.py
.venv/bin/python training/compare_silence_features.py dataset/noise/noise_1.wav
.venv/bin/python training/compare_silence_features.py --synthetic-zeros
```

Or activate the virtual environment first:

```bash
source .venv/bin/activate
python training/compare_silence_features.py
```

## Running the Embedded Project

```bash
cd embedded
source ~/esp/esp-idf/export.sh
idf.py build
idf.py -p /dev/cu.usbmodem101 flash monitor
```

If already flashed and you only want serial output:

```bash
idf.py -p /dev/cu.usbmodem101 monitor
```

## Notes

- `dataset/` is not tracked in git.
- `training/` contains source scripts and notes.
- `final_model/` contains final exported artifacts.
- `embedded/` contains the deployable ESP-IDF project.
