"""Splits data into train/val/test based on video sessions."""
import csv
import shutil
from pathlib import Path

CANDIDATES_DIR = Path("data/candidates")
SELECTED_FILE = Path("data/selected.txt")
DATASET_DIR = Path("data/dataset")

# Change these names to match your actual recording filenames.
# Every session must belong to exactly one split.
SESSION_SPLITS = {
    "cats_001":"train", # Marko
    "cats_002":"train", # Marko
    "cats_003":"train", # Marko
    "cats_004_low_fps":"train",
    "cats_005_low_fps":"train",
    "cats_006_low_fps":"train",
    "cats_007":"train", # light
    "cats_008":"val",
    "cats_009":"test",
    "cats_010":"test",
    "cats_011":"train", # bath
    "cats_012":"train", # Taras
}


def main():
    if not SELECTED_FILE.exists():
        raise FileNotFoundError(SELECTED_FILE)

    selected = [
        line.strip()
        for line in SELECTED_FILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not selected:
        raise ValueError("No selected images found.")

    manifest = []

    for relative_path in selected:
        source = CANDIDATES_DIR / relative_path

        if not source.is_file():
            print(f"WARNING: Missing image: {source}")
            continue

        session = Path(relative_path).parts[0]

        if session not in SESSION_SPLITS:
            raise ValueError(
                f"No split configured for session '{session}'. "
                "Add it to SESSION_SPLITS."
            )

        split = SESSION_SPLITS[session]
        destination_dir = DATASET_DIR / "images" / split
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Prefix with session name to avoid filename collisions.
        destination_name = f"{session}_{source.name}"
        destination = destination_dir / destination_name

        shutil.copy2(source, destination)

        manifest.append({
            "filename": destination_name,
            "source": relative_path,
            "session": session,
            "split": split,
        })

        print(f"{split:5} <- {relative_path}")

    if not manifest:
        raise ValueError("No valid images were copied.")

    manifest_path = DATASET_DIR / "split_manifest.csv"

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "source", "session", "split"],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("\nDataset summary:")
    for split in ("train", "val", "test"):
        count = sum(item["split"] == split for item in manifest)
        print(f"  {split}: {count} images")

    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
