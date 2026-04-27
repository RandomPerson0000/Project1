from ultralytics import YOLO
import torch
import yaml
import os

def main():
    # Tạo file YAML
    data = {
        'path': 'D:\\Do-an1_2classes\\Indoor',
        'train': 'Train/images',
        'val': 'Valid/images',
        'test': 'Test/images',
        'nc': 2,
        'names': ['Fire', 'Not_fire']
    }

    with open('data_2classes.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(" Đã tạo file data_2classes.yaml")

    # Train với GPU
    model = YOLO("yolo26n.pt")

    model.train(
        data='data_2classes.yaml',
        device=0,   # Sử dụng GPU đầu tiên (nếu có nhiều GPU, có thể thay đổi số 0 thành 1, 2, ...)    
        epochs=100, #Quét 100 lần qua toàn bộ dữ liệu huấn luyện
        imgsz=640, 
        batch=-1,   #Số lượng mẫu dữ liệu trong 1 lần học (để -1 thì tự động chọn batch size tối ưu dựa trên VRAM của GPU)
        workers=8,  #Dùng 4 luồng CPU để tải dữ liệu nhanh hơn, giúp tăng tốc độ huấn luyện
        #cache=True, #Lưu trữ dữ liệu vào RAM để tăng tốc độ truy cập trong quá trình huấn luyện (dùng cpu thì không dùng ram)
        amp=True,   #Sử dụng kỹ thuật mixed precision để tăng tốc độ huấn luyện trên GPU hiện đại, giúp giảm sử dụng bộ nhớ và tăng hiệu suất mà không ảnh hưởng đến độ chính xác của mô hình
        patience=20,#Nếu sau 20 epoch liên tiếp mô hình không cải thiện => stop sớm
        save=True,  #Lưu mô hình tốt nhất dựa trên hiệu suất trên tập validation
        save_period=10,  #Lưu mô hình sau mỗi 10 epoch
        project='runs_indoor_2classes/train',
        name='fire_detection_2classes_on_pi5',
        exist_ok=True   # Cho phép ghi đè nếu thư mục đã tồn tại
    )
#Bắt buộc phải có dòng này trên Windows
if __name__ == '__main__':
    main()