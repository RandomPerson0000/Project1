#Kiểm tra dataset
import os
from pathlib import Path

def count_yolo_classes(dataset_path):
    # Cấu trúc thư mục theo mô tả của bạn
    splits = ['train', 'valid', 'test']
    
    # Biến lưu trữ tổng số lượng cho toàn dataset
    total_stats = {
        'only_fire': 0, 
        'only_non_fire': 0, 
        'both': 0, 
        'empty': 0 # Thêm biến đếm ảnh không có label nào (background)
    }

    print(f"Đang kiểm tra dataset tại: {dataset_path}\n")

    for split in splits:
        # Đường dẫn tới thư mục labels: dataset_path/train/labels
        label_dir = Path(dataset_path) / split / 'labels'
        
        if not label_dir.exists():
            print(f"Bỏ qua: Không tìm thấy thư mục {label_dir}")
            continue

        # Thống kê cho từng tập (train/valid/test)
        split_stats = {'only_fire': 0, 'only_non_fire': 0, 'both': 0, 'empty': 0}

        # Đọc tất cả các file .txt trong thư mục labels
        for txt_file in label_dir.glob('*.txt'):
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            # Lấy ra danh sách các class_id có trong file (dùng set để loại bỏ trùng lặp)
            classes_in_image = set()
            for line in lines:
                parts = line.strip().split()
                if parts: # Nếu dòng không trống
                    class_id = int(parts[0]) # Số đầu tiên là class_id
                    classes_in_image.add(class_id)
            
            # Phân loại dựa trên ID đã tìm thấy
            # Theo data.yaml: 0 là fire, 1 là non-fire
            if 0 in classes_in_image and 1 not in classes_in_image:
                split_stats['only_fire'] += 1
            elif 1 in classes_in_image and 0 not in classes_in_image:
                split_stats['only_non_fire'] += 1
            elif 0 in classes_in_image and 1 in classes_in_image:
                split_stats['both'] += 1
            elif len(classes_in_image) == 0:
                split_stats['empty'] += 1 # Ảnh background (file txt rỗng)

        # In kết quả cho từng thư mục
        print(f"--- Thống kê tập [{split.upper()}] ---")
        print(f"  1. Chỉ có nhãn 'fire' (0)          : {split_stats['only_fire']} ảnh")
        print(f"  2. Chỉ có nhãn 'non-fire' (1)      : {split_stats['only_non_fire']} ảnh")
        print(f"  3. Có CẢ HAI nhãn 'fire' & 'non-fire': {split_stats['both']} ảnh")
        if split_stats['empty'] > 0:
            print(f"  *. Không có nhãn nào (Background)  : {split_stats['empty']} ảnh")
        print("-" * 35)

        # Cộng dồn vào tổng
        for key in total_stats:
            total_stats[key] += split_stats[key]

    # In kết quả tổng cộng
    print("\n================ TỔNG CỘNG TOÀN BỘ DATASET ================")
    print(f"  => Tổng số ảnh CHỈ có 'fire'            : {total_stats['only_fire']}")
    print(f"  => Tổng số ảnh CHỈ có 'non-fire'        : {total_stats['only_non_fire']}")
    print(f"  => Tổng số ảnh có CẢ HAI nhãn           : {total_stats['both']}")
    print("===========================================================")

# --- HƯỚNG DẪN SỬ DỤNG ---
# Thay đổi đường dẫn dưới đây thành thư mục gốc chứa file data.yaml và các thư mục train/test/valid
# Ví dụ: DATASET_DIR = r"C:\Users\PC\Downloads\dataset"

DATASET_DIR = r"D:\\Do-an1\\Indoor_merge" 
count_yolo_classes(DATASET_DIR)