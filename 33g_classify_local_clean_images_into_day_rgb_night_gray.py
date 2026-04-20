import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image, ImageFile

# ======================
# PATHS
# ======================
WORKDIR = Path("/media/mkyadava/HD2/walsh-MS-3-Capstone/cnn-models")
MANIFEST_IN = WORKDIR / "detect/day_night/manifests/raw_dump_manifest_50k_clean.csv"
RAW_ROOT = WORKDIR / "detect/day_night/raw_dump/raw"
OUT = WORKDIR / "detect/day_night/manifests/raw_dump_manifest_50k_visual.csv"

# ======================
# CONFIG
# ======================
IMG_SIZE = 128   # small for speed
GRAY_THRESHOLD = 5.0  # channel difference threshold
##
ImageFile.LOAD_TRUNCATED_IMAGES = True
#
bad_files=[]
# ======================
# HELPERS
# ======================

def parse_hour(ts):
    try:
        return pd.to_datetime(ts).hour
    except Exception:
        return None


def get_time_bucket(hour):
    if hour is None:
        return "unknown"
    if hour < 6 or hour >= 18:
        return "night"
    return "day"


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        return np.array(img, dtype=np.float32)
    except Exception:
        return None


def is_gray(img):
    if img is None:
        return False

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    d_rg = np.mean(np.abs(r - g))
    d_rb = np.mean(np.abs(r - b))
    d_gb = np.mean(np.abs(g - b))

    return (d_rg < GRAY_THRESHOLD) and (d_rb < GRAY_THRESHOLD) and (d_gb < GRAY_THRESHOLD)


# ======================
# MAIN
# ======================

def main():
    df = pd.read_csv(MANIFEST_IN, low_memory=False)

    df["image_id"] = df["image_id"].astype(str)

    # Parse time
    df["hour"] = df["datetime"].apply(parse_hour)
    df["time_bucket"] = df["hour"].apply(get_time_bucket)

    visual_modes = []

    print("Classifying images...")

    for i, row in df.iterrows():
        file_name = str(row["file_name"]).strip().replace("\\", "/")
        img_path = RAW_ROOT / file_name

        if not img_path.exists():
            visual_modes.append("missing")
            continue

        tb = row["time_bucket"]

        if tb == "night":
            visual_modes.append("night")
            continue

        # Only check gray for day images
        img = load_image(img_path)

        if img is None:
            bad_files.append(str(img_path))
            visual_modes.append("corrupt")
            continue

        if is_gray(img):
            visual_modes.append("gray")
        else:
            visual_modes.append("day_rgb")

        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(df)}")

    df["visual_mode"] = visual_modes

    df.to_csv(OUT, index=False)

    print("\nSaved:", OUT)
    print("\nCounts:")
    print(df["visual_mode"].value_counts(dropna=False))

    print("\nTime bucket vs visual mode:")
    print(pd.crosstab(df["time_bucket"], df["visual_mode"]))
    #
    bad_out = OUT.parent / "raw_dump_corrupt_or_unreadable_files.txt"
    with open(bad_out, "w") as f:
        for p in bad_files:
            f.write(p + "\n")

    print("Saved bad file list to:", bad_out)


if __name__ == "__main__":
    main()
