import pandas as pd
from pathlib import Path

WORKDIR = Path("/media/mkyadava/HD2/walsh-MS-3-Capstone/cnn-models")
MANIFEST_DIR = WORKDIR / "detect/day_night/manifests"

DAY_IN = MANIFEST_DIR / "manifest_day_all.csv"
NIGHT_IN = MANIFEST_DIR / "manifest_night_all.csv"

OUT_CSV = MANIFEST_DIR / "raw_dump_manifest_50k.csv"

SEED = 42
TOTAL_TARGET = 50000

TARGET_CLASS_QUOTA = 3000
EMPTY_QUOTA = 8000
MIN_OTHER_CLASS_COUNT = 50
MAX_OTHER_CLASS_QUOTA = 500

MINIMUM_TARGET_SPECIES = {
    "panthera onca": "jaguar",
    "puma concolor": "puma",
    "leopardus pardalis": "ocelot",
    "loxodonta africana": "elephant",
    "tayassu pecari": "peccary",
    "aepyceros melampus": "impala",
    "madoqua guentheri": "dik_dik",
    "empty": "empty",
}

TARGET_CLASS_NAMES = {
    "jaguar",
    "puma",
    "ocelot",
    "elephant",
    "peccary",
    "impala",
    "dik_dik",
}

REMOVE_CLASSES = {"unknown", "group", "unknown bird", "vehicle"}


def normalize_file_name(s):
    return str(s).strip().replace("\\", "/").lower()


def main():
    df_day = pd.read_csv(DAY_IN, low_memory=False)
    df_night = pd.read_csv(NIGHT_IN, low_memory=False)

    df = pd.concat([df_day, df_night], ignore_index=True)

    # Remove humans by path
    df["file_name_norm"] = df["file_name"].astype(str).map(normalize_file_name)
    df = df[~df["file_name_norm"].str.startswith("humans/")].copy()

    # Remove noisy classes
    df["selection_class"] = df["selection_class"].astype(str).str.strip().str.lower()
    df = df[~df["selection_class"].isin(REMOVE_CLASSES)].copy()

    # Sequence-safe: one random image per sequence
    df["seq_id"] = df["seq_id"].fillna("NOSEQ_" + df["image_id"].astype(str))
    df = (
        df.groupby("seq_id", group_keys=False)
        .apply(lambda g: g.sample(n=1, random_state=SEED))
        .reset_index(drop=True)
    )

    # Deduplicate image_id just in case
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset=["image_id"], keep="first").copy()

    selected_parts = []

    # A. Empty quota
    df_empty = df[df["is_empty"] == 1].copy()
    if len(df_empty) > 0:
        n = min(len(df_empty), EMPTY_QUOTA)
        selected_parts.append(df_empty.sample(n=n, random_state=SEED))

    # B. Core target species quota
    df_animals = df[df["is_empty"] == 0].copy()

    for cls in sorted(TARGET_CLASS_NAMES):
        sub = df_animals[df_animals["selection_class"] == cls].copy()
        if len(sub) == 0:
            continue
        n = min(len(sub), TARGET_CLASS_QUOTA)
        selected_parts.append(sub.sample(n=n, random_state=SEED))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=df.columns)
    selected_ids = set(selected["image_id"].astype(str)) if not selected.empty else set()

    # C. Other animal classes with at least MIN_OTHER_CLASS_COUNT
    remaining = df[~df["image_id"].astype(str).isin(selected_ids)].copy()
    remaining_animals = remaining[remaining["is_empty"] == 0].copy()

    other_counts = remaining_animals["selection_class"].value_counts()
    eligible_other_classes = [
        cls for cls, cnt in other_counts.items()
        if cnt >= MIN_OTHER_CLASS_COUNT and cls not in TARGET_CLASS_NAMES
    ]

    other_parts = []
    for cls in sorted(eligible_other_classes):
        sub = remaining_animals[remaining_animals["selection_class"] == cls].copy()
        n = min(len(sub), MAX_OTHER_CLASS_QUOTA)
        other_parts.append(sub.sample(n=n, random_state=SEED))

    if other_parts:
        others = pd.concat(other_parts, ignore_index=True)
    else:
        others = pd.DataFrame(columns=df.columns)

    combined = pd.concat([selected, others], ignore_index=True)
    combined = combined.drop_duplicates(subset=["image_id"], keep="first").copy()

    # D. If still below TOTAL_TARGET, fill randomly from remaining pool
    if len(combined) < TOTAL_TARGET:
        used_ids = set(combined["image_id"].astype(str))
        fill_pool = df[~df["image_id"].astype(str).isin(used_ids)].copy()
        need = min(TOTAL_TARGET - len(combined), len(fill_pool))
        if need > 0:
            fill = fill_pool.sample(n=need, random_state=SEED)
            combined = pd.concat([combined, fill], ignore_index=True)

    # Trim if slightly above target
    if len(combined) > TOTAL_TARGET:
        combined = combined.sample(n=TOTAL_TARGET, random_state=SEED).copy()

    combined = combined.drop(columns=["file_name_norm"], errors="ignore").reset_index(drop=True)

    combined["local_relpath"] = combined.apply(
        lambda r: f"detect/day_night/raw_dump/{r['image_id']}_{Path(str(r['file_name'])).name}",
        axis=1,
    )

    combined.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Rows: {len(combined)}")
    print("\nEmpty vs animal:")
    print(combined["is_empty"].value_counts(dropna=False))
    print("\nTop classes:")
    print(combined["selection_class"].value_counts(dropna=False).head(30))
    print("\nUnique locations:", combined["location"].nunique(dropna=True))
    print("Unique seq_id:", combined["seq_id"].nunique(dropna=True))


if __name__ == "__main__":
    main()
