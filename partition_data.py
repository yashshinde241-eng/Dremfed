import os
import shutil
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/raw")          # Where you place HAM10000 files
OUTPUT_DIR  = Path("data/partitions")   # Where silos will be written
METADATA    = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS  = [
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2",
]

LABEL_MAP = {
    "akiec": 0,  # Actinic keratoses
    "bcc":   1,  # Basal cell carcinoma
    "bkl":   2,  # Benign keratosis-like
    "df":    3,  # Dermatofibroma
    "mel":   4,  # Melanoma
    "nv":    5,  # Melanocytic nevi
    "vasc":  6,  # Vascular lesions
}

CLASS_NAMES = {v: k for k, v in LABEL_MAP.items()}


def find_image(image_id: str) -> Path | None:
    """Search for an image file across multiple source directories."""
    for img_dir in IMAGE_DIRS:
        candidate = img_dir / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    return None


def partition_iid(df: pd.DataFrame, n_clients: int) -> list[pd.DataFrame]:
    """IID split – each hospital gets a balanced random subset."""
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return [df.iloc[i::n_clients].reset_index(drop=True) for i in range(n_clients)]


def partition_non_iid(df: pd.DataFrame, n_clients: int) -> list[pd.DataFrame]:
    """
    Non-IID split – each hospital is biased toward certain lesion types,
    simulating real-world specialisation (e.g. melanoma-heavy clinic).
    """
    classes = df["dx"].unique()
    rng = np.random.default_rng(42)

    # Each client gets a Dirichlet-sampled class distribution
    alphas = rng.dirichlet(alpha=[0.5] * n_clients, size=len(classes))
    client_dfs: list[list[pd.DataFrame]] = [[] for _ in range(n_clients)]

    for cls_idx, cls in enumerate(classes):
        cls_df = df[df["dx"] == cls].sample(frac=1, random_state=42)
        splits = (alphas[cls_idx] * len(cls_df)).astype(int)
        # Fix rounding: give remainder to last client
        splits[-1] += len(cls_df) - splits.sum()
        start = 0
        for c, count in enumerate(splits):
            client_dfs[c].append(cls_df.iloc[start : start + count])
            start += count

    return [
        pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
        for parts in client_dfs
    ]


def build_silo(silo_df: pd.DataFrame, silo_path: Path) -> None:
    """Copy images and write a local CSV for one hospital silo."""
    images_path = silo_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    missing = 0
    rows = []
    for _, row in silo_df.iterrows():
        src = find_image(row["image_id"])
        if src is None:
            missing += 1
            continue
        dst = images_path / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        rows.append({
            "image_id":  row["image_id"],
            "label":     LABEL_MAP[row["dx"]],
            "dx":        row["dx"],
            "file_path": str(dst),
        })

    if missing:
        print(f"  ⚠  {missing} images not found – skipped.")

    pd.DataFrame(rows).to_csv(silo_path / "metadata.csv", index=False)
    print(f"  ✓  {len(rows)} images  →  {silo_path}")


def main(n_clients: int = 3, strategy: str = "iid") -> None:
    print(f"\n{'='*60}")
    print(f"  DermFed  ·  Data Partitioning")
    print(f"  Clients : {n_clients}   Strategy : {strategy.upper()}")
    print(f"{'='*60}\n")

    if not METADATA.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at '{METADATA}'.\n"
            "Please follow the README to download HAM10000 and place files correctly."
        )

    df = pd.read_csv(METADATA)
    print(f"Loaded {len(df)} records from metadata.\n")

    # ── Global train / test split (kept consistent across all silos) ─────────
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["dx"], random_state=42)

    # Save global test set (used by the Streamlit dashboard)
    global_test_path = OUTPUT_DIR / "global_test"
    build_silo(test_df, global_test_path)
    print(f"Global test set: {len(test_df)} samples\n")

    # ── Partition training data ──────────────────────────────────────────────
    if strategy == "iid":
        partitions = partition_iid(train_df, n_clients)
    else:
        partitions = partition_non_iid(train_df, n_clients)

    for idx, part_df in enumerate(partitions):
        silo_path = OUTPUT_DIR / f"hospital_{idx}"
        dist = part_df["dx"].value_counts().to_dict()
        print(f"Hospital {idx}  ({len(part_df)} samples)")
        for cls, count in sorted(dist.items()):
            bar = "█" * (count // 20)
            print(f"   {cls:5s}  {count:4d}  {bar}")
        build_silo(part_df, silo_path)
        print()

    # ── Write global label map ───────────────────────────────────────────────
    import json
    label_path = OUTPUT_DIR / "label_map.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        json.dump({"label_map": LABEL_MAP, "class_names": CLASS_NAMES}, f, indent=2)

    print(f"\n{'='*60}")
    print("  Partitioning complete!")
    print(f"  Output  →  {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermFed Data Partitioner")
    parser.add_argument("--n_clients", type=int, default=3,
                        help="Number of hospital silos (default: 3)")
    parser.add_argument("--strategy", choices=["iid", "non_iid"], default="iid",
                        help="Data distribution strategy (default: iid)")
    args = parser.parse_args()
    main(n_clients=args.n_clients, strategy=args.strategy)
