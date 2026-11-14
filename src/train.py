"""YOLO fine-tuning."""

from pathlib import Path

import torch
from ultralytics import YOLO

# Configuration
DATASET = "data/dataset/dataset.yaml"
MODEL = "yolo26n.pt"

EPOCHS = 100
IMAGE_SIZE = 960
BATCH_SIZE = 16


def main():
    # Check hardware
    print("=" * 50)
    print("HARDWARE")
    print("=" * 50)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("Using CPU")
        device = "cpu"

    # Check dataset configuration
    if not Path(DATASET).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET}")

    # Load pretrained model
    print("\nLoading pretrained YOLO model...")
    model = YOLO(MODEL)

    # Fine-tune
    print("\nStarting training...")

    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=20,
        device=device,
        workers=4,
        project="runs/cat_detection",
        name="experiment_01",
        pretrained=True,
        seed=42,
        plots=True,
        save=True,
    )

    best_model = Path(
        "runs/cat_detection/experiment_01/weights/best.pt"
    )

    print("\nTraining finished.")
    print("Best model:", best_model)


if __name__ == "__main__":
    main()
