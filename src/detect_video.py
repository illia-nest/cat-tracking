"""Runs detection on a video stream."""
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "runs/cat_detection/experiment_01/weights/best.pt"

VIDEO_PATH = "data/raw_videos/cats_010.mp4" #9, 10 are in test

OUTPUT_PATH = "runs/cat_detection/detection_output/detection_output_3.mp4" #_1 for 9, _2 for 10

CONFIDENCE = 0.35
IMAGE_SIZE = 960


def main():

    # Select device
    device = 0 if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    # Load trained model
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    # Open input video
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video resolution: {width}x{height}")
    print(f"Video FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")

    # Create output directory
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {OUTPUT_PATH}")

    frame_number = 0

    print("\nProcessing video...\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Run YOLO detection
        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Write annotated frame
        writer.write(annotated_frame)

        frame_number += 1

        # Print progress every 100 frames
        if frame_number % 100 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

    # Release resources
    cap.release()
    writer.release()

    print("\nDetection complete!")
    print("Output saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
