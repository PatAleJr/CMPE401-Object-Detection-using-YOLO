from pathlib import Path


images_dir = Path("../../../ML-Data/VisDrone2019-DET-train/images/")
annotations_dir = Path("../../../ML-Data/VisDrone2019-DET-train/annotations/")

image_files = [
    p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
]
annotation_files = list(annotations_dir.glob("*.txt"))

if not image_files:
    raise FileNotFoundError(f"No images found in {images_dir}")
if not annotation_files:
    raise FileNotFoundError(f"No annotations found in {annotations_dir}")

pair_count = min(len(image_files), len(annotation_files))
mismatches = []

for i in range(pair_count):
    image_stem = image_files[i].stem
    annotation_stem = annotation_files[i].stem
    if image_stem != annotation_stem:
        mismatches.append((i, image_files[i].name, annotation_files[i].name))

print(f"Images: {len(image_files)}")
print(f"Annotations: {len(annotation_files)}")
print(f"Checked aligned indices: {pair_count}")

if len(image_files) != len(annotation_files):
    print("WARNING: Different counts. Some files are unmatched due to length mismatch.")

if not mismatches:
    print("PASS: Sorted image and annotation order is aligned by filename stem.")
else:
    print(f"FAIL: Found {len(mismatches)} mismatches. First 20:")
    for idx, img_name, ann_name in mismatches[:20]:
        print(f"  idx={idx}: image={img_name} annotation={ann_name}")
