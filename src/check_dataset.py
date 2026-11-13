"""Checks the correspondance between the number of images and the number of labels."""
from pathlib import Path

DATASET_DIR = Path("data/dataset")
SPLITS = ["train", "val", "test"]
VALID_CLASSES = {0, 1}

errors = []

for split in SPLITS:
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ] if image_dir.exists() else []

    labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []

    print(f"\n{split}: {len(images)} images, {len(labels)} label files")

    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels}

    for orphan in label_stems - image_stems:
        errors.append(f"{split}: label has no matching image: {orphan}")

    for image in images:
        label = label_dir / f"{image.stem}.txt"

        # Missing label file is allowed: it may be a negative image.
        if not label.exists():
            continue

        for line_number, line in enumerate(
            label.read_text().splitlines(), start=1
        ):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label}:{line_number}: expected 5 values"
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{label}:{line_number}: invalid numeric value"
                )
                continue

            if class_id not in VALID_CLASSES:
                errors.append(
                    f"{label}:{line_number}: invalid class {class_id}"
                )

            if not all(
                0 <= value <= 1
                for value in (x, y, width, height)
            ):
                errors.append(
                    f"{label}:{line_number}: coordinates out of range"
                )

            if width <= 0 or height <= 0:
                errors.append(
                    f"{label}:{line_number}: invalid box dimensions"
                )

if errors:
    print("\nDataset problems:")
    for error in errors:
        print(" -", error)
else:
    print("\nBasic dataset checks passed.")
