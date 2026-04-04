from pathlib import Path
import cv2


DATASET_ROOT = Path("../../../ML-Data/VisDrone2019-DET-val").resolve()
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
YOLO_LABELS_DIR = DATASET_ROOT / "labels"


def parse_visdrone_boxes(annotation_path: Path):
    boxes = []
    with annotation_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue

            x, y, w, h, score, cls_id, trunc, occ = map(float, parts[:8])
            if int(score) == 0:
                continue
            if int(cls_id) <= 0:
                continue

            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)
            boxes.append((int(cls_id), x1, y1, x2, y2))
    return boxes


def visdrone_boxes_to_yolo_lines(boxes, image_width: int, image_height: int):
    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0

        xc /= float(image_width)
        yc /= float(image_height)
        bw /= float(image_width)
        bh /= float(image_height)

        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)

        yolo_cls = cls_id - 1
        lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def parse_yolo_boxes(label_path: Path, image_width: int, image_height: int):
    boxes = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            x1 = (xc - bw / 2.0) * image_width
            y1 = (yc - bh / 2.0) * image_height
            x2 = (xc + bw / 2.0) * image_width
            y2 = (yc + bh / 2.0) * image_height
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def draw_boxes(image, boxes, color, prefix: str):
    out = image.copy()
    for cls_id, x1, y1, x2, y2 in boxes:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(
            out,
            f"{prefix} c{cls_id}",
            (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def convert_all_annotations():
    YOLO_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    converted = 0

    for image_path in image_files:
        ann_path = ANNOTATIONS_DIR / f"{image_path.stem}.txt"
        if not ann_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        vis_boxes = parse_visdrone_boxes(ann_path)
        yolo_lines = visdrone_boxes_to_yolo_lines(vis_boxes, w, h)

        out_label_path = YOLO_LABELS_DIR / f"{image_path.stem}.txt"
        with out_label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    return converted, image_files


def create_visual_check(image_files):
    for image_path in image_files:
        ann_path = ANNOTATIONS_DIR / f"{image_path.stem}.txt"
        yolo_label_path = YOLO_LABELS_DIR / f"{image_path.stem}.txt"
        if not ann_path.exists() or not yolo_label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        vis_boxes = parse_visdrone_boxes(ann_path)
        yolo_boxes = parse_yolo_boxes(yolo_label_path, w, h)

        vis_img = draw_boxes(image, vis_boxes, (0, 255, 0), "VIS")
        yolo_img = draw_boxes(image, yolo_boxes, (255, 0, 0), "YOLO")

        cv2.putText(vis_img, "Original VisDrone Annotation", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(yolo_img, "Converted YOLO Annotation", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        side_by_side = cv2.hconcat([vis_img, yolo_img])

        vis_path = Path("visdrone_original_boxes.jpg")
        yolo_path = Path("visdrone_yolo_boxes.jpg")
        compare_path = Path("visdrone_conversion_compare.jpg")

        cv2.imwrite(str(vis_path), vis_img)
        cv2.imwrite(str(yolo_path), yolo_img)
        cv2.imwrite(str(compare_path), side_by_side)

        return image_path, vis_path.resolve(), yolo_path.resolve(), compare_path.resolve(), len(vis_boxes), len(yolo_boxes)

    raise RuntimeError("Could not find any image with both source and converted labels.")


def main():
    converted_count, image_files = convert_all_annotations()
    print(f"Converted annotation files: {converted_count}")
    print(f"YOLO labels directory: {YOLO_LABELS_DIR}")

    sample_image, vis_path, yolo_path, compare_path, vis_n, yolo_n = create_visual_check(image_files)
    print(f"Sample image: {sample_image}")
    print(f"VIS boxes drawn: {vis_n}")
    print(f"YOLO boxes drawn: {yolo_n}")
    print(f"Saved original-annotation image: {vis_path}")
    print(f"Saved converted-yolo image: {yolo_path}")
    print(f"Saved side-by-side comparison: {compare_path}")


if __name__ == "__main__":
    main()
