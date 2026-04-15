from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

RUN_NAME = "evaluation-01" # Where you want the results to be stored
DATASET_YAML = "visdrone.yaml"
DATASET_SPLIT = "eval"  # test-dev split key in visdrone.yaml
WEIGHTS = "runs/detect/train_visdrone_yolo26n-03/weights/best.pt"

def main() -> None:
    print(f"Python executable: {Path(torch.__file__).resolve().parents[2] / 'python.exe'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")

    # if not torch.cuda.is_available():
    #     raise RuntimeError(
    #         "CUDA is not available in this Python environment. "
    #         "Run with venv\\Scripts\\python.exe or activate the venv first."
    #     )

    # device = "cuda:0"
    # print(f"Using device: {device}. CUDA device: {torch.cuda.get_device_name(0)}")

    model = YOLO(WEIGHTS)
    print(f"Done training. Starting validation on '{DATASET_SPLIT}' split...")

    # Run a final validation pass to ensure confusion matrix and summary artifacts are saved.
    model.val(
        data=DATASET_YAML,
        split=DATASET_SPLIT,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
    )

    print("Done validation.")

if __name__ == "__main__":
    main()
