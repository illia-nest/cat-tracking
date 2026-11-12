"""Video recording and frame extraction."""

import csv
from pathlib import Path

import cv2
import imagehash
from PIL import Image

VIDEO_DIR = Path("data/raw_videos")
OUTPUT_DIR = Path("data/candidates")
SHEET_DIR = Path("data/contact_sheets")

# Extract one frame every 1 second.
SAMPLE_INTERVAL_SECONDS = 1

# Frames with a perceptual hash distance <= this value
# are treated as near-duplicates within the same recording.
PHASH_DISTANCE_THRESHOLD = 5

# Contact sheet settings.
THUMBNAIL_SIZE = (240, 135)
SHEET_COLUMNS = 5


def make_contact_sheets(image_paths, session_name):
    """Create contact sheets with filenames under each thumbnail."""
    if not image_paths:
        return

    from PIL import ImageDraw

    cell_width = 260
    cell_height = 170
    rows_per_sheet = 30
    sheet_size = SHEET_COLUMNS * rows_per_sheet

    for start in range(0, len(image_paths), sheet_size):
        batch = image_paths[start:start + sheet_size]
        rows = (len(batch) + SHEET_COLUMNS - 1) // SHEET_COLUMNS

        sheet = Image.new(
            "RGB",
            (SHEET_COLUMNS * cell_width, rows * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for index, path in enumerate(batch):
            image = Image.open(path).convert("RGB")
            image.thumbnail(THUMBNAIL_SIZE)

            x = (index % SHEET_COLUMNS) * cell_width
            y = (index // SHEET_COLUMNS) * cell_height

            sheet.paste(image, (x, y))
            draw.text(
                (x, y + 140),
                path.name,
                fill="black",
            )

        part = start // sheet_size + 1
        output_path = SHEET_DIR / f"{session_name}_{part:02d}.jpg"
        sheet.save(output_path, quality=90)
        print(f"Contact sheet: {output_path}")


def extract_video(video_path):
    session_name = video_path.stem
    session_output = OUTPUT_DIR / session_name
    session_output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        print(f"ERROR: Invalid FPS in {video_path}")
        capture.release()
        return

    duration = frame_count / fps
    interval_frames = max(1, round(fps * SAMPLE_INTERVAL_SECONDS))

    print(
        f"\nProcessing {video_path.name}: "
        f"{duration:.1f}s, {fps:.1f} FPS"
    )

    saved_paths = []
    hashes = []
    metadata = []

    frame_index = 0
    candidate_index = 0
    saved_index = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_index % interval_frames == 0:
            candidate_index += 1

            # Convert OpenCV BGR to RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            current_hash = imagehash.phash(pil_image)

            # Compare only against previously saved frames
            # from this recording session.
            is_duplicate = any(
                current_hash - old_hash
                <= PHASH_DISTANCE_THRESHOLD
                for old_hash in hashes
            )

            timestamp = frame_index / fps

            if not is_duplicate:
                filename = (
                    f"{session_name}_"
                    f"{timestamp:08.2f}s.jpg"
                )
                output_path = session_output / filename

                pil_image.save(output_path, quality=95)

                hashes.append(current_hash)
                saved_paths.append(output_path)
                saved_index += 1

                metadata.append({
                    "session": session_name,
                    "filename": filename,
                    "timestamp_seconds": round(timestamp, 2),
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                })

        frame_index += 1

    capture.release()

    # Save extraction metadata.
    csv_path = session_output / "metadata.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "session",
                "filename",
                "timestamp_seconds",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata)

    print(
        f"Extracted {saved_index} unique frames "
        f"from {video_path.name}"
    )

    make_contact_sheets(saved_paths, session_name)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        print(f"No MP4 files found in {VIDEO_DIR}")
        return

    for video_path in videos:
        extract_video(video_path)

    print("\nFrame extraction complete.")


if __name__ == "__main__":
    main()
