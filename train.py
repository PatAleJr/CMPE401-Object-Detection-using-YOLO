from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO


RUN_NAME = "train_visdrone_yolo26n-01"
DATASET_YAML = "visdrone.yaml"
WEIGHTS = "yolo26n.pt"
EPOCHS = 50
IMG_SIZE = 512


def main() -> None:
    print(f"Python executable: {Path(torch.__file__).resolve().parents[2] / 'python.exe'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "Run with venv\\Scripts\\python.exe or activate the venv first."
        )

    device = "cuda:0"
    print(f"Using device: {device}. CUDA device: {torch.cuda.get_device_name(0)}")

    model = YOLO(WEIGHTS)

    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
    )

    print("Done training. Starting validation...")

    # Run a final validation pass to ensure confusion matrix and summary artifacts are saved.
    model.val(
        data=DATASET_YAML,
        device=device,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
    )

    print("Done validation.")

if __name__ == "__main__":
    main()
