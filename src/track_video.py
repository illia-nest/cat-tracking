
"""Video tracking with BoT-SORT and track confirmation."""

from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from tracking import PersistentIdentityManager, TrackConfirmation

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "runs/cat_detection/experiment_01/weights/best.pt"

# VIDEO_PATH = "data/raw_videos/cats_009.mp4"
VIDEO_PATH = "data/raw_videos/cats_010.mp4"

EXP_NO = 6

OUTPUT_PATH = (
    f"runs/cat_detection/tracking_output_01/"
    f"tracking_output_0{EXP_NO}_b_identity_manager.mp4"
)

CONFIDENCE = 0.10
IMAGE_SIZE = 960

TRACKER = "configs/botsort_cats.yaml"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # Select device
    device = 0 if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    # Load trained detector
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    # Open input video
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")

    # Prepare output directory
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Initialize track confirmation
    confirmation = TrackConfirmation(
        confirmation_window=20,
        required_observations=10,
        min_confidence=0.30,
        max_missed_during_confirmation=4,
        max_missed_after_confirmation=60,
    )
    
    identity_manager = PersistentIdentityManager(
        min_observations=12,
        min_class_ratio=0.85,
        owner_timeout_frames=90,
    )
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot create output video: {OUTPUT_PATH}"
        )

    # Statistics
    frame_number = 0

    raw_ids_seen = set()
    confirmed_ids_seen = set()

    print("\nStarting BoT-SORT tracking...\n")

    # --------------------------------------------------
    # Video processing loop
    # --------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Run detection and tracking
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False,
        )

        result = results[0]


        # Collect raw BoT-SORT IDs.
        if result.boxes is not None and result.boxes.id is not None:
            raw_ids_seen.update(
                result.boxes.id.int().cpu().tolist()
            )

        # Update confirmation exactly once per frame.
        confirmed_ids = confirmation.update(result)

        confirmed_ids_seen.update(confirmed_ids)

        # Update persistent identity manager.
        identity_by_track = identity_manager.update(
            result,
            confirmed_ids,
            frame_number,
        )

        # Keep only detections with a confirmed persistent identity.
        if result.boxes is not None and result.boxes.id is not None:

            raw_ids = result.boxes.id.int().cpu().tolist()

            keep = torch.tensor(
                [
                    track_id in identity_by_track
                    for track_id in raw_ids
                ],
                dtype=torch.bool,
                device=result.boxes.id.device,
            )

            filtered_result = result[keep]

            if (
                filtered_result.boxes is not None
                and filtered_result.boxes.id is not None
            ):

                original_ids = (
                    filtered_result.boxes.id.int().cpu().tolist()
                )

                # Stable display ID:
                # Marko = 1
                # Taras = 2
                identity_display_ids = {
                    "marko": 1,
                    "taras": 2,
                }

                box_data = filtered_result.boxes.data.clone()

                display_ids = torch.tensor(
                    [
                        identity_display_ids[identity_by_track[track_id]]
                        for track_id in original_ids
                    ],
                    dtype=box_data.dtype,
                    device=box_data.device,
                )

                # Ultralytics tracked box layout:
                # x1, y1, x2, y2, track_id, confidence, class_id
                box_data[:, 4] = display_ids

                filtered_result = Results(
                    orig_img=filtered_result.orig_img,
                    path=filtered_result.path,
                    names=filtered_result.names,
                    boxes=box_data,
                )

            annotated_frame = filtered_result.plot()

        else:
            annotated_frame = result.plot()


        # --------------------------------------------------
        # Write output frame
        # --------------------------------------------------

        writer.write(annotated_frame)

        frame_number += 1

        if frame_number % 100 == 0:

            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()
    writer.release()

    # --------------------------------------------------
    # Final statistics
    # --------------------------------------------------
    print("\nTracking statistics")
    print("------------------------------")

    print(f"Raw BoT-SORT IDs: {len(raw_ids_seen)}")
    print(f"Confirmed IDs:    {len(confirmed_ids_seen)}")

    print("\nPersistent identity assignments:")

    for track_id, identity in sorted(
        identity_manager.track_identity.items()
    ):
        print(f"  BoT-SORT ID {track_id} -> {identity.upper()}")

    print("------------------------------")


if __name__ == "__main__":
    main()
