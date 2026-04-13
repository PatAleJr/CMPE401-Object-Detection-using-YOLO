from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


DEFAULT_WEIGHTS = "runs/detect/train_visdrone_yolov3u/weights/best.pt"
DEFAULT_DATA_YAML = "visdrone.yaml"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_IMG_SIZE = 640
DEFAULT_SEED = 401
DEFAULT_WARMUP = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_val_images(data_yaml_path: Path) -> list[Path]:
	with data_yaml_path.open("r", encoding="utf-8") as f:
		data_cfg = yaml.safe_load(f)

	if not isinstance(data_cfg, dict) or "val" not in data_cfg:
		raise ValueError(f"Could not find 'val' entry in {data_yaml_path}")

	val_path = Path(str(data_cfg["val"]))
	if not val_path.is_absolute():
		val_path = (data_yaml_path.parent / val_path).resolve()

	if not val_path.exists():
		raise FileNotFoundError(f"Validation image directory does not exist: {val_path}")

	images = [
		p
		for p in val_path.rglob("*")
		if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
	]
	if not images:
		raise RuntimeError(f"No validation images found in: {val_path}")
	return images


def synchronize_if_cuda(device: str) -> None:
	if device.startswith("cuda"):
		torch.cuda.synchronize()


def main() -> None:
	weights_path = Path(DEFAULT_WEIGHTS).resolve()
	data_yaml_path = Path(DEFAULT_DATA_YAML).resolve()

	if not weights_path.exists():
		raise FileNotFoundError(f"Weights not found: {weights_path}")
	if not data_yaml_path.exists():
		raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")
	if DEFAULT_SAMPLE_SIZE <= 0:
		raise ValueError("--sample-size must be > 0")

	all_val_images = resolve_val_images(data_yaml_path)
	sample_count = min(DEFAULT_SAMPLE_SIZE, len(all_val_images))

	rng = random.Random(DEFAULT_SEED)
	sampled_images = rng.sample(all_val_images, sample_count)

	# device = "cuda:0" if torch.cuda.is_available() else "cpu"
	device = "cpu"
	model = YOLO(str(weights_path))

	print(f"Model: {weights_path}")
	print(f"Validation images found: {len(all_val_images)}")
	print(f"Sampled images: {sample_count}")
	print(f"Device: {device}")
	print(f"Image size: {DEFAULT_IMG_SIZE}")

	warmup_count = min(DEFAULT_WARMUP, sample_count)
	for image_path in sampled_images[:warmup_count]:
		model.predict(source=str(image_path), imgsz=DEFAULT_IMG_SIZE, device=device, verbose=False)

	inference_times_ms: list[float] = []
	for image_path in sampled_images:
		synchronize_if_cuda(device)
		start = time.perf_counter()
		model.predict(source=str(image_path), imgsz=DEFAULT_IMG_SIZE, device=device, verbose=False)
		synchronize_if_cuda(device)
		end = time.perf_counter()
		inference_times_ms.append((end - start) * 1000.0)

	avg_ms = statistics.mean(inference_times_ms)
	median_ms = statistics.median(inference_times_ms)
	min_ms = min(inference_times_ms)
	max_ms = max(inference_times_ms)
	throughput = 1000.0 / avg_ms if avg_ms > 0 else float("inf")

	print("\nInference timing summary")
	print(f"Average: {avg_ms:.2f} ms/image")
	print(f"Median : {median_ms:.2f} ms/image")
	print(f"Min    : {min_ms:.2f} ms/image")
	print(f"Max    : {max_ms:.2f} ms/image")
	print(f"Throughput (approx): {throughput:.2f} images/sec")


if __name__ == "__main__":
	main()
