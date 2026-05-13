"""
Record raw WAV clips into dataset/<class>/.

Workflow (dataset rebuild): 1) run this script (per class) -> 2) preprocess_audio.py
-> 3) augment_audio.py -> 4) retrain (e.g. speech_training.py).
"""
import os
import re
import sounddevice as sd
from scipy.io.wavfile import write

# Settings (match ESP32 + ML expectations)
SAMPLE_RATE = 16000  # 16 kHz

# ---------------------------------------------------------------------------
# Rebuild plan — single place for how many clips and how long (per class).
# `count` = new files to add this round; `duration_sec` = mic capture length
# (preprocess_audio.py will still make fixed 1.0 s training clips).
# ---------------------------------------------------------------------------
REBUILD_PLAN = [
    # class_name      count  seconds  note
    ("backward",       30,  3.0,  "+30 clips; 3.0 s; index auto from last backward_N.wav."),
    ("spottieottie",   30,  3.0,  "+30 clips; 3.0 s; long phrase, then 1.0 s in preprocess."),
    ("noise",          30,  2.5,  "+30 clips; background / room tone."),
    ("unknown_words",  30,  2.0,  "+30 clips; non-keyword speech; preprocess -> 1.0 s."),
]


def _print_rebuild_plan(base_path: str) -> None:
    """Show exactly what the script expects you to record for this rebuild."""
    print()
    print("=" * 72)
    print("  DATASET REBUILD — what to record (run this script once per class)")
    print("=" * 72)
    for word, count, sec, note in REBUILD_PLAN:
        print(
            f"\n  • {word}\n"
            f"      Record {count} new clips, {sec} s each, into:\n"
            f"      {os.path.join(base_path, word)}/\n"
            f"      ({note})"
        )
    print(
        "\n"
        "  Un-comment exactly ONE `record_one_planned_class(...)` call at the bottom\n"
        "  to start that class (start index is auto from existing {word}_N.wav files).\n"
        "=" * 72
        + "\n"
    )


def next_available_index(word: str, class_folder: str) -> int:
    """
    Scan class_folder for files named {word}_<n>.wav and return max(n)+1, or 1 if none.
    """
    if not os.path.isdir(class_folder):
        return 1
    pat = re.compile(rf"^{re.escape(word)}_(\d+)\.wav$", re.IGNORECASE)
    best = 0
    for f in os.listdir(class_folder):
        m = pat.match(f)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def record_samples(
    word: str,
    num_samples: int,
    save_path: str,
    duration_sec: float,
    start_index=None,
):
    """
    Record WAV clips. Filenames: {word}_{start}.wav ... {word}_{start+num_samples-1}.wav
    If start_index is None, the next free index is chosen from existing files in save_path.
    """
    os.makedirs(save_path, exist_ok=True)
    start = start_index if start_index is not None else next_available_index(word, save_path)
    end_num = start + num_samples - 1

    print(
        f"\n--- '{word}' ---\n"
        f"  duration: {duration_sec} s  |  count: {num_samples}  |  "
        f"files: {word}_{start}.wav … {word}_{end_num}.wav\n"
        f"  save: {os.path.abspath(save_path)}"
    )
    print("Press ENTER to start each recording.\n")

    n_samples = int(duration_sec * SAMPLE_RATE)
    for i in range(num_samples):
        n = start + i
        input(f"[{i + 1}/{num_samples}] ENTER to record '{word}' clip #{n} ({duration_sec}s)...")

        print("Recording...")
        audio = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()

        filename = os.path.join(save_path, f"{word}_{n}.wav")
        write(filename, SAMPLE_RATE, audio)

        print(f"Saved: {filename}\n")

    print(f"Done recording '{word}'.\n")


def record_one_planned_class(class_name: str, base: str) -> bool:
    """Look up `class_name` in REBUILD_PLAN and run record_samples. Returns True if run."""
    for word, count, sec, _note in REBUILD_PLAN:
        if word == class_name:
            record_samples(
                word,
                count,
                os.path.join(base, word),
                sec,
                start_index=None,
            )
            return True
    print(f"Unknown class {class_name!r}. Choose one of: { [w for w, *_ in REBUILD_PLAN] }")
    return False


if __name__ == "__main__":
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE = os.path.join(_repo_root, "dataset")
    print(f"Dataset directory: {os.path.abspath(BASE)}")

    print(
        "\nWorkflow: record (this script) -> preprocess_audio.py -> augment_audio.py -> retrain.\n"
    )
    _print_rebuild_plan(BASE)

    # Set to one of: "backward", "spottieottie", "noise", "unknown_words"
    # (must match a name in REBUILD_PLAN). None = only print the plan, no capture.
    RECORDING_CLASS = "noise"

    if RECORDING_CLASS is not None:
        record_one_planned_class(RECORDING_CLASS, BASE)
    else:
        print(
            "Not recording yet. Set RECORDING_CLASS = \"backward\" (or the next class) "
            "in this `if __name__` block, then run again.\n"
        )
