"""Model evaluation script."""

import torch
from ultralytics import YOLO


def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(
        "runs/cat_detection/experiment_01/weights/best.pt"
    )

    metrics = model.val(
        data="data/dataset/dataset.yaml",
        split="test",
        imgsz=960,
        device=device,
        plots=True,
    )

    print("\nTEST RESULTS")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    main()
