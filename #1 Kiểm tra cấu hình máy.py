import torch
import os
import platform

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

# Kiểm tra số lượng GPU khả dụng
num_gpus = torch.cuda.device_count()
print(f"Số lượng GPU đang có: {num_gpus}")
# In tên của GPU (nếu có)
if num_gpus > 0:
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Không tìm thấy GPU nào (Đang chạy bằng CPU).")

print("="*40)
print("THÔNG TIN CPU (CƠ BẢN)")
print("="*40)

# Tên CPU
print(f"Tên bộ vi xử lý : {platform.processor()}")

# Hệ điều hành chi tiết
print(f"Hệ điều hành     : {platform.system()} {platform.release()}")

# Số luồng (Logical Processors)
# Đây chính là con số tối đa bạn có thể dùng cho tham số 'workers'
cpu_threads = os.cpu_count()
print(f"Tổng số luồng    : {cpu_threads}")
print("="*40)