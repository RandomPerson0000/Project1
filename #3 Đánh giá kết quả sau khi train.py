#Đánh giá kết quả sau khi train
from ultralytics import YOLO

def main():
    # Load model đã train
    model = YOLO("D:\\Do-an1\\Trained_model\\runs\\detect\\runs_indoor_fire_with_negatives_samples\\train\\fire_detection_fire_with_negatives_samples_on_pi5\\weights\\best.pt")

    # Validate
    metrics = model.val(
        data='data_fire_with_negatives_samples.yaml',
        device=0
    )
    print(f"\n Kết quả Validate:")
    print(f"   mAP50    : {metrics.box.map50:.3f}")
    print(f"   mAP50-95 : {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall   : {metrics.box.mr:.3f}")
    # Đánh giá
    print(f"\n Đánh giá:")
    if metrics.box.map50 >= 0.9:
        print("Rất tốt!")
    elif metrics.box.map50 >= 0.7:
        print("Tốt!")
    elif metrics.box.map50 >= 0.5:
        print("Trung bình - Cần train thêm")
    else:
        print("Chưa tốt - Cần xem lại dataset")

if __name__ == '__main__':
    main()