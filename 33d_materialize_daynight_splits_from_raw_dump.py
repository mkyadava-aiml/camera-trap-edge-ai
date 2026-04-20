import shutil
from pathlib import Path
import pandas as pd

WORKDIR = Path("/media/mkyadava/HD2/walsh-MS-3-Capstone/cnn-models")
MANIFEST_DIR = WORKDIR / "detect/day_night/manifests"
RAW_ROOT = WORKDIR / "detect/day_night/raw_dump/raw"
DATA_ROOT = WORKDIR / "detect/day_night/data"

MANIFEST_MAP = {
    ("day", "train"): MANIFEST_DIR / "manifest_day_train.csv",
    ("day", "val"): MANIFEST_DIR / "manifest_day_val.csv",
    ("day", "test"): MANIFEST_DIR / "manifest_day_test.csv",
    ("night", "train"): MANIFEST_DIR / "manifest_night_train.csv",
    ("night", "val"): MANIFEST_DIR / "manifest_night_val.csv",
    ("night", "test"): MANIFEST_DIR / "manifest_night_test.csv",
}


def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        return "exists"

    try:
        dst.hardlink_to(src)
        return "linked"
    except Exception:
        try:
            shutil.copy2(src, dst)
            return "copied"
        except Exception:
            return "failed"


def main():
    total_missing = 0
    total_done = 0

    for (time_bucket, split_name), manifest_path in MANIFEST_MAP.items():
        if not manifest_path.exists():
            print(f"[WARN] Missing manifest: {manifest_path}")
            continue

        df = pd.read_csv(manifest_path, low_memory=False)
        out_dir = DATA_ROOT / time_bucket / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        missing = 0
        done = 0

        print(f"\nProcessing {time_bucket}/{split_name} ...")

        for _, row in df.iterrows():
            file_name = str(row["file_name"]).strip().replace("\\", "/")
            image_id = str(row["image_id"])

            src = RAW_ROOT / file_name
            dst = out_dir / f"{image_id}_{Path(file_name).name}"

            if not src.exists():
                missing += 1
                continue

            status = safe_link_or_copy(src, dst)
            if status in ("linked", "copied", "exists"):
                done += 1
            else:
                missing += 1

        print(f"{time_bucket}/{split_name} complete")
        print(f"Materialized: {done}")
        print(f"Missing: {missing}")

        total_done += done
        total_missing += missing

    print("\nAll done.")
    print(f"Total materialized: {total_done}")
    print(f"Total missing: {total_missing}")


if __name__ == "__main__":
    main()
