"""Prepares data generated from preannotate.py for importing by CVAT."""
import zipfile
from pathlib import Path

PREANNOTATIONS_DIR = Path("data/preannotations")
OUTPUT_DIR = Path("data/cvat_import")

SPLITS = ["train", "val", "test"]


def create_archive(split):
    source_dir = PREANNOTATIONS_DIR / split
    labels_dir = source_dir / "labels"

    if not labels_dir.exists():
        print(f"Skipping {split}: no labels directory")
        return

    output_zip = OUTPUT_DIR / f"cat_preannotations_{split}.zip"
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))

    # CVAT YOLO 1.1 archive structure.
    archive_prefix = "obj_train_data"

    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:

        # One temporary class.
        archive.writestr("obj.names", "cat\n")

        # Dataset configuration.
        archive.writestr(
            "obj.data",
            "classes = 1\n"
            "names = obj.names\n"
            "train = train.txt\n"
        )

        image_list = []

        for label_file in label_files:
            # The corresponding image filename.
            # Images are already uploaded to the CVAT task.
            image_extensions = [".jpg", ".jpeg", ".png"]

            image_name = None

            for extension in image_extensions:
                candidate = (
                    Path("data/dataset/images")
                    / split
                    / f"{label_file.stem}{extension}"
                )

                if candidate.exists():
                    image_name = candidate.name
                    break

            if image_name is None:
                print(f"WARNING: No image found for {label_file.name}")
                continue

            # CVAT uses the image filename to match annotations.
            image_list.append(
                f"{archive_prefix}/{image_name}"
            )

            archive.write(
                label_file,
                f"{archive_prefix}/{label_file.name}"
            )

        archive.writestr(
            "train.txt",
            "\n".join(image_list) + "\n"
        )

    print(f"Created: {output_zip}")
    print(f"Annotation files: {len(label_files)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        create_archive(split)

    print("\nCVAT archives ready.")


if __name__ == "__main__":
    main()
