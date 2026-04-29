#Dựa trên dataset có fire và not_fire đều đã label, phân loại thành 2 file riêng biệt
import os
import shutil
from pathlib import Path
import yaml

# =========================
# CẤU HÌNH
# =========================
SOURCE_DATASET = Path(r"D:\\Do-an1\\Indoor_merge")
TARGET_DATASET = Path(r"D:\\Do-an1\\Samples")

SPLITS = ["Train", "Valid", "Test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class mapping dataset cũ
OLD_FIRE_CLASS_ID = 0
OLD_NOT_FIRE_CLASS_ID = 1

def is_image_file(filename):
    return Path(filename).suffix.lower() in IMAGE_EXTS

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def process_label_file(src_label_path: Path, dst_label_path: Path):
    """
    Giữ lại chỉ các dòng thuộc class Fire (class 0).
    Xóa toàn bộ Not_fire (class 1).
    Nếu không còn dòng nào thì tạo file rỗng.
    """
    kept_lines = []

    if src_label_path.exists():
        with open(src_label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                # bỏ qua dòng lỗi format
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            # Chỉ giữ Fire
            if class_id == OLD_FIRE_CLASS_ID:
                # Dataset mới vẫn chỉ có 1 class Fire nên class id vẫn là 0
                parts[0] = "0"
                kept_lines.append(" ".join(parts))

    # Ghi file label mới
    with open(dst_label_path, "w", encoding="utf-8") as f:
        if kept_lines:
            f.write("\n".join(kept_lines) + "\n")
        else:
            # tạo file rỗng cho negative image
            f.write("")

def convert_split(split_name):
    src_images_dir = SOURCE_DATASET / split_name / "images"
    src_labels_dir = SOURCE_DATASET / split_name / "labels"

    dst_images_dir = TARGET_DATASET / split_name / "images"
    dst_labels_dir = TARGET_DATASET / split_name / "labels"

    ensure_dir(dst_images_dir)
    ensure_dir(dst_labels_dir)

    if not src_images_dir.exists():
        print(f"[WARNING] Không tìm thấy thư mục ảnh: {src_images_dir}")
        return

    image_files = [f for f in os.listdir(src_images_dir) if is_image_file(f)]

    total_images = 0
    fire_images = 0
    negative_images = 0

    for image_name in image_files:
        total_images += 1

        src_image_path = src_images_dir / image_name
        dst_image_path = dst_images_dir / image_name

        label_name = Path(image_name).stem + ".txt"
        src_label_path = src_labels_dir / label_name
        dst_label_path = dst_labels_dir / label_name

        # Copy ảnh
        shutil.copy2(src_image_path, dst_image_path)

        # Xử lý label
        process_label_file(src_label_path, dst_label_path)

        # Thống kê
        with open(dst_label_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content:
            fire_images += 1
        else:
            negative_images += 1

    print(f"\n[INFO] Split: {split_name}")
    print(f" - Tổng số ảnh      : {total_images}")
    print(f" - Ảnh có Fire      : {fire_images}")
    print(f" - Ảnh Negative     : {negative_images}")

def create_yaml():
    data = {
        "path": str(TARGET_DATASET),
        "train": "Train/images",
        "val": "Valid/images",
        "test": "Test/images",
        "nc": 1,
        "names": ["Fire"]
    }

    yaml_path = TARGET_DATASET / "data_fire_with_negatives_samples.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n[DONE] Đã tạo file YAML: {yaml_path}")

def main():
    print("[START] Đang chuyển dataset 2 class -> 1 class Fire...")

    for split in SPLITS:
        convert_split(split)

    create_yaml()

    print("\n[COMPLETE] Hoàn tất chuyển dataset.")
    print(f"Dataset mới nằm tại: {TARGET_DATASET}")

if __name__ == "__main__":
    main()