"""
Create augmented copies for selected classes (per-class target counts).

Workflow: after record_audio.py and preprocess_audio.py, run this, then retrain.

Uses random sampling of *existing* clips and mild transforms. Does not augment every file.
"""
import os
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
SEED = 42
# Output names: {word}_aug_r{SEED}_{n:04d}_{tag}.wav

# Only these classes are augmented, with the given targets (0 = not run in __main__).
# spottieottie, unknown_words, noise: no augmentation (target 0).
AUGMENT_TARGETS = {
    "backward": 40,
    "spottieottie": 0,
    "noise": 0,
    "unknown_words": 0,
}


def _repo_dataset_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "dataset")


def _force_length(y, length):
    """Center-pad or trim to exactly `length` samples."""
    if len(y) == length:
        return y.astype(np.float32, copy=False)
    if len(y) < length:
        pad = length - len(y)
        pl, pr = pad // 2, pad - pad // 2
        return np.pad(y, (pl, pr)).astype(np.float32)
    return y[:length].astype(np.float32)


def _augment_speech_like(audio, sr, rng):
    """Mild pitch / time / gain / light noise. Returns (output, short_tag)."""
    length = len(audio)
    choice = rng.integers(0, 7)
    if choice == 0:
        y = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1.0)
        tag = "p1"
    elif choice == 1:
        y = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1.0)
        tag = "m1"
    elif choice == 2:
        y = librosa.effects.time_stretch(audio, rate=1.07)
        tag = "f07"
    elif choice == 3:
        y = librosa.effects.time_stretch(audio, rate=0.93)
        tag = "s93"
    elif choice == 4:
        y = audio * 0.88
        tag = "g88"
    elif choice == 5:
        y = audio * 1.12
        tag = "g112"
    else:
        n = rng.normal(0.0, 0.0025, size=audio.shape)
        y = np.clip(audio + n.astype(np.float32), -1.0, 1.0)
        tag = "hn"
    y = _force_length(np.asarray(y, dtype=np.float32), length)
    return y, tag


def _augment_noise_like(audio, sr, rng):
    """Gain, mild time stretch, light hiss — no strong pitch shift."""
    length = len(audio)
    choice = rng.integers(0, 5)
    if choice == 0:
        y = audio * 0.82
        tag = "g82"
    elif choice == 1:
        y = audio * 1.18
        tag = "g118"
    elif choice == 2:
        y = librosa.effects.time_stretch(audio, rate=1.05)
        tag = "f05"
    elif choice == 3:
        y = librosa.effects.time_stretch(audio, rate=0.95)
        tag = "s95"
    else:
        n = rng.normal(0.0, 0.004, size=audio.shape)
        y = np.clip(audio + n.astype(np.float32), -1.0, 1.0)
        tag = "hs"
    y = _force_length(np.asarray(y, dtype=np.float32), length)
    return y, tag


def _list_source_wavs(class_dir, prefix):
    """Original clips only (skip augmented outputs). Any base name like prefix_*.wav counts."""
    if not os.path.isdir(class_dir):
        return []
    out = []
    pre = prefix + "_"
    for f in os.listdir(class_dir):
        if not f.endswith(".wav"):
            continue
        if "_aug_" in f:
            continue
        if f.startswith(pre):
            out.append(os.path.join(class_dir, f))
    return sorted(out)


def augment_class_folder(class_name, dataset_dir, target_count, rng, use_noise_style):
    class_dir = os.path.join(dataset_dir, class_name)
    sources = _list_source_wavs(class_dir, class_name)
    if not sources:
        print(f"  {class_name}: no source .wav files found, skipping.")
        return 0

    created = 0
    attempts = 0
    max_attempts = max(200, target_count * 200)

    while created < target_count and attempts < max_attempts:
        attempts += 1
        path = sources[int(rng.integers(0, len(sources)))]
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if len(audio) < 100:
            continue

        if use_noise_style:
            y, tag = _augment_noise_like(audio, sr, rng)
        else:
            y, tag = _augment_speech_like(audio, sr, rng)

        fname = f"{class_name}_aug_r{SEED}_{created + 1:04d}_{tag}.wav"
        out_path = os.path.join(class_dir, fname)
        if os.path.isfile(out_path):
            continue

        sf.write(out_path, y, SAMPLE_RATE)
        created += 1
        print(f"  Saved: {fname}")

    return created


if __name__ == "__main__":
    dataset = _repo_dataset_dir()
    print(f"Dataset: {os.path.abspath(dataset)}")
    print(
        "Workflow: record -> preprocess_audio.py -> this script -> retrain.\n"
        f"Per-class targets (seed={SEED}): {AUGMENT_TARGETS}\n"
    )

    rng = np.random.default_rng(SEED)

    # noise_style=True for background-like; speech keywords use False
    style_by_class = {
        "backward": False,
        "spottieottie": False,
        "noise": True,
        "unknown_words": False,
    }

    summary = {}
    for class_name, target in AUGMENT_TARGETS.items():
        if target <= 0:
            print(f"\n{class_name}: skipped (augmentation disabled for this class).")
            summary[class_name] = 0
            continue
        use_noise = style_by_class.get(class_name, False)
        print(f"\n{class_name} (target={target}, noise_style={use_noise}):")
        n = augment_class_folder(class_name, dataset, target, rng, use_noise)
        summary[class_name] = n
        print(f"  -> created {n} files (target {target})")

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v} augmented clips")
    print("Done.\n")
