"""Runs YOLO26n to preannotate data with the cat class."""
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = "yolo26n.pt"

INPUT_DIR = Path("data/dataset/images")
OUTPUT_DIR = Path("data/preannotations")

SPLITS = ["train", "val", "test"]

# Start permissive. We will correct false positives in CVAT.
CONFIDENCE = 0.20
IMAGE_SIZE = 960

# Use "0" for the first CUDA GPU, or "cpu".
DEVICE = "0"


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Find the cat class ID instead of hardcoding it.
    cat_ids = [
        class_id
        for class_id, name in model.names.items()
        if name.lower() == "cat"
    ]

    if not cat_ids:
        raise RuntimeError(
            "The loaded model does not have a COCO cat class."
        )

    cat_class_id = cat_ids[0]

    print(f"Cat class ID: {cat_class_id}")
    print(f"Cat class name: {model.names[cat_class_id]}")

    for split in SPLITS:
        image_dir = INPUT_DIR / split

        if not image_dir.exists():
            print(f"Skipping missing directory: {image_dir}")
            continue

        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not images:
            print(f"No images in {split}")
            continue

        output_dir = OUTPUT_DIR / split
        preview_dir = output_dir / "previews"
        labels_dir = output_dir / "labels"

        preview_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        csv_rows = []

        print(f"\nProcessing {split}: {len(images)} images")

        for index, image_path in enumerate(images, start=1):

            results = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                classes=[cat_class_id],
                device=DEVICE,
                verbose=False,
            )

            result = results[0]

            # Read original image dimensions.
            height, width = result.orig_shape

            # Prepare YOLO-format annotations.
            annotation_lines = []

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().tolist()
                    )

                    confidence = float(
                        box.conf[0].cpu().item()
                    )

                    # Clip coordinates to image boundaries.
                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))

                    box_width = x2 - x1
                    box_height = y2 - y1

                    if box_width <= 0 or box_height <= 0:
                        continue

                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height
                    normalized_width = box_width / width
                    normalized_height = box_height / height

                    # Temporary class ID 0 = cat.
                    annotation_lines.append(
                        f"0 {center_x:.6f} "
                        f"{center_y:.6f} "
                        f"{normalized_width:.6f} "
                        f"{normalized_height:.6f}"
                    )

                    csv_rows.append({
                        "image": image_path.name,
                        "class": "cat",
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    })

            # Save annotations, including empty files
            # for images where no cats were detected.
            label_path = labels_dir / f"{image_path.stem}.txt"
            label_path.write_text(
                "\n".join(annotation_lines)
                + ("\n" if annotation_lines else "")
            )

            # Save visualization.
            preview = result.plot(
                labels=True,
                conf=True,
                line_width=2,
            )

            preview_path = preview_dir / image_path.name
            cv2.imwrite(str(preview_path), preview)

            if index % 25 == 0:
                print(f"Processed {index}/{len(images)}")

        # Save confidence and bounding box information.
        csv_path = output_dir / "detections.csv"

        with csv_path.open("w", newline="") as file:
            fieldnames = [
                "image", "class", "confidence",
                "x1", "y1", "x2", "y2",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Finished {split}")
        print(f"Previews: {preview_dir}")
        print(f"Annotations: {labels_dir}")
        print(f"Detections: {len(csv_rows)}")

    print("\nPre-annotation complete!")


if __name__ == "__main__":
    main()
