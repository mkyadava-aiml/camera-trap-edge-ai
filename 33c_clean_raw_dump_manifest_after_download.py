import pandas as pd
from pathlib import Path

MANIFEST = Path("/media/mkyadava/HD2/walsh-MS-3-Capstone/cnn-models/detect/day_night/manifests/raw_dump_manifest_50k.csv")
RAW_ROOT = Path("/media/mkyadava/HD2/walsh-MS-3-Capstone/cnn-models/detect/day_night/raw_dump/raw")

OUT = MANIFEST.parent / "raw_dump_manifest_50k_clean.csv"

df = pd.read_csv(MANIFEST, low_memory=False)


def exists(row):
    p = RAW_ROOT / str(row["file_name"]).strip().replace("\\", "/")
    return p.exists()


df["exists"] = df.apply(exists, axis=1)
clean_df = df[df["exists"]].copy()
clean_df.drop(columns=["exists"], inplace=True)

clean_df.to_csv(OUT, index=False)

print("Original:", len(df))
print("Clean:", len(clean_df))
print()

print("By class:")
print(clean_df["selection_class"].value_counts(dropna=False).head(30))
print()

print("Empty vs animal:")
print(clean_df["is_empty"].value_counts(dropna=False))
