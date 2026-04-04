from ultralytics import YOLOE
from pathlib import Path
import cv2


model = YOLOE("yolo26n.pt")


images_dir = Path("../../../ML-Data/VisDrone2019-DET-train/images/")
annotations_dir = Path("../../../ML-Data/VisDrone2019-DET-train/annotations/")

image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]


result = model([image_files[100]])
result[0].show()