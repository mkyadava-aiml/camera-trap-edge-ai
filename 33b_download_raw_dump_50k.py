from pathlib import Path
from urllib.parse import quote
import concurrent.futures as cf
import requests
import pandas as pd

MANIFEST = Path("/media/mkyadava/HD2/walsh-MS-3-Capstone/cnn-models/detect/day_night/manifests/raw_dump_manifest_50k.csv")
OUT_ROOT = Path("/media/mkyadava/HD2/walsh-MS-3-Capstone/cnn-models/detect/day_night/raw_dump")
RAW_ROOT = OUT_ROOT / "raw"

BASE_URLS = [
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped",
    "https://storage.googleapis.com/public-datasets-lila/wcs-unzipped",
]

MAX_WORKERS = 16
TIMEOUT = 60

RAW_ROOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST, low_memory=False)


def download_one(file_name: str):
    rel_path = Path(str(file_name).strip().replace("\\", "/"))
    out_path = RAW_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return ("exists", file_name)

    quoted = "/".join(quote(part) for part in rel_path.parts)

    for base in BASE_URLS:
        url = f"{base}/{quoted}"
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                if r.status_code != 200:
                    continue

                tmp_path = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

                tmp_path.replace(out_path)
                return ("downloaded", file_name)

        except Exception:
            continue

    return ("failed_all_sources", file_name)


def main():
    file_names = (
        df["file_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .tolist()
    )

    counts = {}
    failures = []
    i = 0

    print("Download starting...")

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for status, file_name in ex.map(download_one, file_names):
            counts[status] = counts.get(status, 0) + 1
            i += 1

            if i % 50 == 0:
                print("Completed:", i)

            if status not in ("downloaded", "exists"):
                failures.append((status, file_name))

    print("\nDownload summary:")
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")

    fail_csv = OUT_ROOT / "raw_dump_download_failures.csv"
    if failures:
        pd.DataFrame(failures, columns=["status", "file_name"]).to_csv(fail_csv, index=False)
        print(f"Failures saved to: {fail_csv}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()
