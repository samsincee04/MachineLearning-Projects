#!/usr/bin/env python3
# Copyright: usage script for the keyword-spotter project.
#
# PURPOSE (embedded vs training feature comparison)
# -------------------------------------------------
# We check whether **embedded** silence (or near-silence) INT8 features are distributed
# similarly to **Python / training** features produced by the same microfrontend
# op + the same int8 quant formula as `embedded/main/main.cpp`.
#
# * If embedded features are much hotter, more saturated, or strongly shifted positive
#   relative to this script's output, the likely problem is **frontend / pipeline
#   mismatch** on device (state, scaling, or audio).
# * If feature statistics are similar but on-device predictions are still wrong,
#   the likely locus moves toward **model behavior / calibration** (or label mapping),
#   not the feature magnitudes per se.
#
# The training Keras model consumes **float** spectrograms; TFLM on device uses
# **int8** with the hand-written quant in main.cpp. This script emulates that int8
# path from **uint16** microfrontend output so log lines are comparable to
# FEATURE_SUMMARY / FEATURE_HEAD / FEATURE_SLICE from the firmware.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DATASET = _REPO_ROOT / "dataset"
_VENV_PY = _REPO_ROOT / ".venv" / "bin" / "python3"

# Match `training/speech_training.py` + `embedded/main/main.cpp` microfrontend defaults.
I16_MAX = 2**15 - 1
I16_MIN = -2**15
FSAMP = 16000
WAVE_LEN = 16000
WINDOW_SIZE_MS = 64
WINDOW_STEP_MS = 48
NUM_FILTERS = 32

# Same as C: kFeatureValueScale / kFeatureValueDiv, then v -= 128.
K_FEATURE_VALUE_SCALE = 256
K_FEATURE_VALUE_DIV = 666

try:
    import tensorflow as tf
    from tensorflow.lite.experimental.microfrontend.python.ops import (  # type: ignore
        audio_microfrontend_op as frontend_op,
    )
except ImportError as e:  # pragma: no cover
    hint = ""
    if _VENV_PY.is_file():
        hint = (
            f"\n  This repo has a venv with TensorFlow. From the repo root run:\n"
            f"    {_VENV_PY} training/compare_silence_features.py [args...]\n"
            f"  or: source .venv/bin/activate  &&  python training/compare_silence_features.py [args...]\n"
        )
    else:
        hint = (
            "\n  Install TensorFlow (with the microfrontend op), e.g. in a venv:\n"
            "    python3 -m venv .venv && source .venv/bin/activate\n"
            "    pip install 'tensorflow<2.19'  # or the version you use for speech_training.py\n"
        )
    print(
        "compare_silence_features: need TensorFlow and the tflite microfrontend op.\n"
        f"  ({e})"
        f"{hint}",
        file=sys.stderr,
    )
    sys.exit(1)


def _decode_wav_int16_16000(wav_path: Path) -> tuple[np.ndarray, str]:
    """Match training: tf.audio.decode_wav (float) then int16 scale as in get_spectrogram."""
    raw = tf.io.read_file(str(wav_path))
    audio, _ = tf.audio.decode_wav(raw)
    w = tf.squeeze(audio, axis=-1)
    w = w[:WAVE_LEN]
    pad = WAVE_LEN - tf.shape(w)[0]
    w = tf.pad(w, [[0, pad]], mode="CONSTANT", constant_values=0.0)
    i16 = tf.cast(0.5 * w * (I16_MAX - I16_MIN), tf.int16)
    return i16.numpy().astype(np.int16), str(wav_path)


def _synthetic_int16_16000(zeros: bool) -> np.ndarray:
    if zeros:
        return np.zeros(WAVE_LEN, dtype=np.int16)
    # Soft synthetic "silence" ~ training create_silence_dataset (low RMS white noise in float);
    # then convert as get_spectrogram: float* scale -> int16. Keep RMS small.
    rng = np.random.default_rng(0)
    f32 = 0.02 * rng.standard_normal(WAVE_LEN).astype(np.float32)
    i16 = (0.5 * f32 * (I16_MAX - I16_MIN)).astype(np.int32)
    return np.clip(i16, I16_MIN, I16_MAX).astype(np.int16)


def _microfrontend_u16(
    int16_1d: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Run the same `audio_microfrontend` path as `get_spectrogram` in speech_training,
    with explicit kwargs to align with embedded C FrontendConfig in main.cpp.
    Returns uint16 (T, 32) with each row a time frame; T should be 20 for 1 s at 16 kHz.
    """
    x = tf.constant(int16_1d, dtype=tf.int16)
    spec = frontend_op.audio_microfrontend(
        x,
        sample_rate=FSAMP,
        window_size=WINDOW_SIZE_MS,
        window_step=WINDOW_STEP_MS,
        num_channels=NUM_FILTERS,
        lower_band_limit=125.0,
        upper_band_limit=7500.0,
        smoothing_bits=10,
        even_smoothing=0.025,
        odd_smoothing=0.06,
        min_signal_remaining=0.05,
        enable_pcan=True,
        pcan_strength=0.95,
        pcan_offset=80.0,
        gain_bits=21,
        enable_log=True,
        scale_shift=6,
        out_type=tf.uint16,
    )
    a = spec.numpy()
    # Doc: each row = time frame, each column = channel.
    note = f"shape={a.shape} dtype={a.dtype}"
    if a.shape == (NUM_FILTERS, 20) or (a.ndim == 2 and a.shape[0] == 32 and a.shape[1] == 20):
        a = a.T
        note += " (transposed to time x channel)"
    return a, note


def int8_from_u16_embed_style(
    u16: np.ndarray,
) -> tuple[np.ndarray, int, int, int, int, float, float]:
    """(int8[...], n_lo, n_hi, min, max, mean, avg_abs) — same as embedded clamp accounting."""
    u = u16.astype(np.int64, copy=False)
    raw = (u * K_FEATURE_VALUE_SCALE + (K_FEATURE_VALUE_DIV // 2)) // K_FEATURE_VALUE_DIV
    v = raw - 128
    n_lo = int(np.sum(v < -128))
    n_hi = int(np.sum(v > 127))
    v_clipped = np.clip(v, -128, 127).astype(np.int8)
    flat = v_clipped.ravel()
    n = flat.size
    mn = int(flat.min())
    mx = int(flat.max())
    mean = float(flat.sum(dtype=np.int64) / n)
    avg_abs = float(np.abs(flat.astype(np.int64)).sum() / n)
    return v_clipped, n_lo, n_hi, mn, mx, mean, avg_abs


def _print_py_from_int8(
    file_tag: str, int8_20x32: np.ndarray, u16_for_clamps: np.ndarray
) -> None:
    """int8_20x32: row-major (time, mel); u16: same shape for clamp counts (pre-clip)."""
    if int8_20x32.shape != (20, 32):
        print(
            f"PY_WARN expected int8 shape (20, 32) for slice loop, got {int8_20x32.shape}.",
            file=sys.stderr,
        )
    # Summaries on final int8; clamp counts from pre-clip (must use u16 path).
    flat = int8_20x32.ravel()
    n = flat.size
    mean = float(flat.sum(dtype=np.int64) / n)
    avg_abs = float(np.abs(flat.astype(np.int64)).sum() / n)
    mn, mx = int(flat.min()), int(flat.max())
    u = u16_for_clamps.astype(np.int64, copy=False)
    raw = (u * K_FEATURE_VALUE_SCALE + (K_FEATURE_VALUE_DIV // 2)) // K_FEATURE_VALUE_DIV
    v = raw - 128
    n_lo = int(np.sum(v < -128))
    n_hi = int(np.sum(v > 127))

    safe = str(file_tag).replace("\n", " ")
    print(
        f"PY_FEATURE_SUMMARY file={safe} min={mn} max={mx} mean={mean:.4f} avg_abs={avg_abs:.4f} "
        f"clamp_lo={n_lo} clamp_hi={n_hi}"
    )
    head = ",".join(str(int(x)) for x in flat[:32].tolist())
    print(f"PY_FEATURE_HEAD file={safe} values={head}")
    nrows = int(int8_20x32.shape[0])
    for s in range(nrows):
        row = int8_20x32[s, :32]
        rm = int(row.min())
        rx = int(row.max())
        sm = float(row.sum(dtype=np.int64) / 32)
        saa = float(np.abs(row.astype(np.int64)).sum() / 32)
        print(
            f"PY_FEATURE_SLICE file={safe} slice={s} min={rm} max={rx} mean={sm:.4f} avg_abs={saa:.4f}"
        )


def _select_quietest_wav(root: Path) -> Path | None:
    wavs = list(root.rglob("*.wav"))
    if not wavs:
        return None
    best: tuple[float, Path] | None = None
    for p in wavs:
        try:
            raw = tf.io.read_file(str(p))
            audio, _ = tf.audio.decode_wav(raw)
            w = tf.squeeze(audio, axis=-1)[:WAVE_LEN].numpy()
            rms = float(np.sqrt(np.mean(w**2)))
        except (tf.errors.NotFoundError, OSError, ValueError):
            continue
        if best is None or rms < best[0]:
            best = (rms, p)
    return best[1] if best else None


def _default_input_path() -> tuple[Path, str]:
    """Prefer a path suggesting silence; else the quietest WAV under dataset/."""
    if _DATASET.is_dir():
        for sub in _DATASET.iterdir():
            if sub.is_dir() and "silence" in sub.name.lower():
                w = sorted(sub.glob("*.wav"))
                if w:
                    return w[0], f"picked: first in silence-like folder {sub.name}"
        silence_named = [p for p in _DATASET.rglob("*.wav") if "silence" in p.stem.lower()]
        if silence_named:
            return silence_named[0], "picked: filename contains 'silence'"
        q = _select_quietest_wav(_DATASET)
        if q is not None:
            return q, "note: no obvious silence file; using quietest WAV in dataset/ by RMS"
    raise FileNotFoundError(
        f"No default WAV under {_DATASET}; pass --wav or use --synthetic-zeros / --synthetic-noise"
    )


def _resolve_wav_arg(p: str | Path) -> Path:
    """
    Relative paths like dataset/foo.wav are usually written from repo root. If the
    shell cwd is training/, that same string would wrongly point at training/dataset/.
    Prefer cwd if the file exists there; else try under the repo root.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.is_file():
        return from_cwd
    from_repo = (_REPO_ROOT / path).resolve()
    if from_repo.is_file():
        return from_repo
    return from_cwd


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare training-style microfrontend + INT8 stats to embedded FEATURE_* logs."
    )
    p.add_argument("wav", nargs="?", help="16 kHz mono WAV; if omitted, auto-select (see help)")
    p.add_argument(
        "--synthetic-zeros",
        action="store_true",
        help="Use 1 s of int16 silence (no file)",
    )
    p.add_argument(
        "--synthetic-noise",
        action="store_true",
        help="Use 1 s of low-level synthetic noise (int16) instead of a file",
    )
    args = p.parse_args()

    if sum(bool(x) for x in (args.synthetic_zeros, args.synthetic_noise, args.wav)) > 1:
        print("Use only one of: positional wav path, --synthetic-zeros, --synthetic-noise", file=sys.stderr)
        return 2

    if args.synthetic_zeros:
        w_i16 = _synthetic_int16_16000(zeros=True)
        tag = f"synthetic_zeros@{_REPO_ROOT}"
    elif args.synthetic_noise:
        w_i16 = _synthetic_int16_16000(zeros=False)
        tag = f"synthetic_low_noise@{_REPO_ROOT}"
    elif args.wav:
        wp = _resolve_wav_arg(args.wav)
        w_i16, _ = _decode_wav_int16_16000(wp)
        tag = str(wp.as_posix())
    else:
        path, how = _default_input_path()
        w_i16, _ = _decode_wav_int16_16000(path)
        print(f"PY_INFO {how} -> {path.as_posix()}", flush=True)
        tag = str(path.as_posix())

    u16, shape_note = _microfrontend_u16(w_i16)
    print(f"PY_INFO u16 {shape_note}", flush=True)
    if u16.shape != (20, 32):
        print(
            f"PY_WARN expected microfrontend (20, 32), got {u16.shape}; adjust script if needed.",
            file=sys.stderr,
        )

    int8_f, _, _, _, _, _, _ = int8_from_u16_embed_style(u16)
    _print_py_from_int8(tag, int8_f, u16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
