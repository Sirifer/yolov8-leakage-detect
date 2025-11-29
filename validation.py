from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("runs/detect/train9/weights/best.pt") 
# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics = model.val(
    data="/cms/user/huangsuyun/dataset/data.yaml",
    conf=0.05,         # 更低的置信度阈值
    iou=0.8,          # 更高的NMS IOU
    imgsz=1024,       # 更大分辨率
    agnostic_nms=True,
    max_det=1000
)
metrics = model.val(save_json=True) # save validation results to JSON

# Print metrics
print("mAP@0.5:0.95 =", metrics.box.map)
print("mAP@0.5      =", metrics.box.map50)
print("Precision    =", metrics.box.mp)
print("Recall       =", metrics.box.mr)
print("F1 Score     =", metrics.box.f1)
