#Kiểm tra dataset có đồng nhất, cùng kiểu định dạng annotations 
from pathlib import Path

DATASET_PATH = Path(r"D:\\Do-an1\\Samples")
SPLITS = ["Train", "Valid", "Test"]

for split in SPLITS:
    labels_dir = DATASET_PATH / split / "labels"
    print(f"\n=== Checking {split} ===")

    for txt_file in labels_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) > 5:
                print(f"[SEGMENT?] {txt_file} - line {i}: {len(parts)} values")
                break

#Chuyển segmented sang bounding box
from pathlib import Path

DATASET_PATH = Path(r"D:\\Do-an1\\Samples")
SPLITS = ["Train", "Valid", "Test"]

def polygon_to_bbox(parts):
    """
    parts: ['class', x1, y1, x2, y2, x3, y3, ...]
    Tất cả tọa độ đang là normalized (0-1)
    """
    class_id = parts[0]
    coords = list(map(float, parts[1:]))

    xs = coords[0::2]
    ys = coords[1::2]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return f"{class_id} {x_center} {y_center} {width} {height}"

for split in SPLITS:
    labels_dir = DATASET_PATH / split / "labels"
    print(f"\n=== Processing {split} ===")

    for txt_file in labels_dir.glob("*.txt"):
        new_lines = []

        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) == 0:
                continue

            if len(parts) == 5:
                # detect format chuẩn
                new_lines.append(" ".join(parts))

            elif len(parts) > 5 and len(parts[1:]) % 2 == 0:
                # segmentation -> convert sang bbox
                bbox_line = polygon_to_bbox(parts)
                new_lines.append(bbox_line)

            else:
                # format lỗi, bỏ qua
                print(f"[SKIP] Format lỗi trong file: {txt_file}")

        with open(txt_file, "w", encoding="utf-8") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
            else:
                f.write("")