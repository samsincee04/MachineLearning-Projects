from pathlib import Path
import random
import shutil

random.seed(42)

ROOT = Path("training/dataset")
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"

classes = ["shoe", "no_shoe"]
exts = {".jpg", ".jpeg", ".png", ".bmp"}

# Clear old processed split if it exists
if PROCESSED.exists():
    shutil.rmtree(PROCESSED)

for cls in classes:
    files = sorted([p for p in (RAW / cls).iterdir() if p.suffix.lower() in exts])

    if len(files) == 0:
        raise ValueError(f"No images found in {RAW/cls}")

    random.shuffle(files)

    n = len(files)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, split_files in splits.items():
        out_dir = PROCESSED / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in split_files:
            shutil.copy2(f, out_dir / f.name)

    print(f"{cls}: total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("Dataset split complete.")
for split in ["train", "val", "test"]:
    for cls in classes:
        count = len(list((PROCESSED / split / cls).iterdir()))
        print(f"{split}/{cls}: {count}")
