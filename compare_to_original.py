from ultralytics import YOLOE
from pathlib import Path
import cv2

model = YOLOE("runs/detect/train_visdrone_yolo26n-01/weights/best.pt")

images_dir = Path("../../../ML-Data/VisDrone2019-DET-val/images/")
annotations_dir = Path("../../../ML-Data/VisDrone2019-DET-val/annotations/")

image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

def render_overlay(image_path: Path):
    def parse_visdrone_annotations(annotation_path: Path):
        gt_boxes = []
        with annotation_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                values = [v.strip() for v in line.split(",")]
                if len(values) < 8:
                    continue

                x, y, w, h, score, cls_id, trunc, occ = map(float, values[:8])
                # In VisDrone DET annotations, score == 0 indicates ignored regions.
                if int(score) == 0:
                    continue

                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                gt_boxes.append((x1, y1, x2, y2, int(cls_id), int(occ), int(trunc)))

        return gt_boxes

    annotation_path = annotations_dir / f"{image_path.stem}.txt"
    gt_boxes = parse_visdrone_annotations(annotation_path)

    result = model([image_path])[0]
    image = result.plot()

    for x1, y1, x2, y2, cls_id, occ, trunc in gt_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"GT c{cls_id}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(image, "GT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Prediction", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    output_path = Path("stage1_pretrained_vs_gt.jpg")
    cv2.imwrite(str(output_path), image)

    print(f"Image: {image_path}")
    print(f"Annotation: {annotation_path}")
    print(f"GT boxes: {len(gt_boxes)}")
    print(f"Saved overlay: {output_path.resolve()}")
    return

img = images_dir / "0000023_01233_d_0000011.jpg"
result = model([img])
result[0].show()
render_overlay(img)