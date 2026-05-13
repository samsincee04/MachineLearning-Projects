"""
Preprocess raw recordings into fixed 1.0 s, 16 kHz mono clips for training.

Recordings are often *longer* at collection time (see record_audio.py). This step trims
(optional) silence, then pads or crops to the fixed 1-second training format.
"""
import os
import librosa
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 1.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)


def process_audio(file_path, trim_silence=True):
    # Load audio as mono at 16 kHz
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Normalize volume only if nonzero
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    # Trim silence for spoken-word folders, but not for noise/background
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=20)

    # Pad or trim to fixed 1-second length
    if len(audio) < TARGET_LENGTH:
        pad_amount = TARGET_LENGTH - len(audio)
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        audio = np.pad(audio, (pad_left, pad_right))
    else:
        audio = audio[:TARGET_LENGTH]

    return audio


def process_folder(folder_path, trim_silence=True):
    if not os.path.isdir(folder_path):
        print(f"Skipping missing folder: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)

            try:
                audio = process_audio(file_path, trim_silence=trim_silence)
                sf.write(file_path, audio, SAMPLE_RATE)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(_repo_root, "dataset")
    print(f"Dataset: {os.path.abspath(base_path)}")
    print(
        "Preprocess: in-place 1.0 s @ 16 kHz. Raw clips may be longer; trim/pad to training length.\n"
    )

    # trim_silence: True for speech-like folders, False for noise
    folder_settings = {
        "backward": True,
        "spottieottie": True,
        "unknown_words": True,
        "noise": False,
    }

    for subfolder, trim_silence in folder_settings.items():
        folder = os.path.join(base_path, subfolder)
        print(f"\nProcessing {subfolder}...")
        process_folder(folder, trim_silence=trim_silence)

    print(
        "\nAll audio processed. Next: augment_audio.py, then retrain (speech_training.py).\n"
    )
