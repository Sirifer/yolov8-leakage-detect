import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# 加载你训练好的模型（替换为你的模型路径）
model = YOLO('runs/detect/train9/weights/best.pt')

# 创建保存图像的目录
save_dir = "detected_glue"
os.makedirs(save_dir, exist_ok=True)

# 打开默认摄像头（0表示第一个摄像头，如果不对可以尝试1或2）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 放大图像 4 倍
    frame = cv2.resize(frame, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_LINEAR)

    # 推理
    results = model(frame)

    # 可视化检测结果（直接在图像上绘制框和标签）
    annotated_frame = results[0].plot()

    # 检查是否检测到 glue（类别名）
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        class_ids = boxes.cls.cpu().numpy()  # 类别索引
        names = results[0].names              # 类别名称映射字典
        for class_id in class_ids:
            class_name = names[int(class_id)]
            if class_name == "glue":
                # 保存该帧图像（加时间戳）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{save_dir}/glue_detected_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[INFO] 检测到 glue，图像已保存到：{filename}")
                break  # 一帧只保存一次

    # 显示图像
    cv2.imshow('Module Defect Detection', annotated_frame)

    # 按q键退出
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
