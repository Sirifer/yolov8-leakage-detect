

##################################################分成三类
import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
#        配置路径
# ==========================
MODEL_PATH = "/cms/user/huangsuyun/runs/detect/train6/weights/best.pt"

IMAGE_DIR = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/images/test")
LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/labels/test")

OUTPUT_DIR = Path("runs/detect/bfbonding/predict/test_iou5_conf10")
RESULT_DIR = Path("runs/detect/bfbonding/results_iou5_conf10")  
REPORT_FILE = "misclassified_pollutant_glue_background.csv"

CLASSES = ["glue", "pollutant", "background"]  # 0 / 1 / 2

# ==========================
# 创建目录
# ==========================
for c in [
    "glue_to_pollutant", "glue_to_bkg",
    "pollutant_to_glue", "pollutant_to_bkg",
    "bkg_to_glue", "bkg_to_pollutant",
    "correct_glue", "correct_pollutant", "correct_bkg",
]:
    (RESULT_DIR / c).mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
#       加载模型
# ==========================
model = YOLO(MODEL_PATH)

print("开始预测 test 数据集 ...")

results = model.predict(
    source=str(IMAGE_DIR),
    save=True,
    save_txt=True,
    project=str(OUTPUT_DIR.parent),
    name=OUTPUT_DIR.name,
    exist_ok=True,
    conf=0.1,
    iou=0.5
)

# 预测标签目录
labels_pred_dir = OUTPUT_DIR / "labels"

y_true, y_pred = [], []
records = []

possible_exts = [".bmp", ".BMP", ".Bmp", ".png", ".PNG"]

# ==========================
#        逐张处理
# ==========================
for label_file in LABEL_DIR.glob("*.txt"):

    # ---- 真实标签 ----
    if label_file.stat().st_size == 0:
        true_cls = 2  # background
    else:
        with open(label_file) as f:
            ids = [int(x.split()[0]) for x in f.readlines()]
        if 0 in ids:
            true_cls = 0
        elif 1 in ids:
            true_cls = 1
        else:
            true_cls = 2

    # ---- 找图片 ----
    img_path = None
    for ext in possible_exts:
        p = IMAGE_DIR / (label_file.stem + ext)
        if p.exists():
            img_path = p
            break
    if img_path is None:
        print(f"⚠️ 找不到图片：{label_file.stem}")
        continue

    # ---- 预测标签 ----
    pred_file = labels_pred_dir / label_file.name
    if not pred_file.exists() or pred_file.stat().st_size == 0:
        pred_cls = 2  # background
    else:
        with open(pred_file) as f:
            pred_ids = [int(x.split()[0]) for x in f.readlines()]
        if 0 in pred_ids:
            pred_cls = 0
        elif 1 in pred_ids:
            pred_cls = 1
        else:
            pred_cls = 2

    y_true.append(true_cls)
    y_pred.append(pred_cls)

    # ==========================
    #    分类保存预测图
    # ==========================
    pred_img = OUTPUT_DIR / img_path.name
    if not pred_img.exists():
        pred_img = OUTPUT_DIR / (img_path.stem + ".jpg")

    if pred_img.exists():

        # glue 类
        if true_cls == 0:
            if pred_cls == 0:
                tgt = "correct_glue"
            elif pred_cls == 1:
                tgt = "glue_to_pollutant"
            else:
                tgt = "glue_to_bkg"

        # pollutant 类
        elif true_cls == 1:
            if pred_cls == 1:
                tgt = "correct_pollutant"
            elif pred_cls == 0:
                tgt = "pollutant_to_glue"
            else:
                tgt = "pollutant_to_bkg"

        # background 类
        else:
            if pred_cls == 0:
                tgt = "bkg_to_glue"
            elif pred_cls == 1:
                tgt = "bkg_to_pollutant"
            else:
                tgt = "correct_bkg"

        shutil.copy(pred_img, RESULT_DIR / tgt / pred_img.name)

    records.append({
        "image": str(img_path),
        "true": CLASSES[true_cls],
        "pred": CLASSES[pred_cls],
    })

# ==========================
# 保存记录
# ==========================
df = pd.DataFrame(records)
df.to_csv(REPORT_FILE, index=False)
print(f"✅ CSV 结果已保存：{REPORT_FILE}")

# ==========================
#     3×3 混淆矩阵
# ==========================
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
print("\n混淆矩阵：\n", cm)

print("\n分类报告：\n")
print(classification_report(
    y_true, y_pred,
    labels=[0,1,2],
    target_names=CLASSES
))

# ==========================
#    绘制混淆矩阵
# ==========================
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    square=True, linewidths=0.5
)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.title("Confusion Matrix (3 Classes)", fontsize=16)
plt.tight_layout()

conf_matrix_path = OUTPUT_DIR / "confusion_matrix_3class.png"
plt.savefig(conf_matrix_path, dpi=300)
plt.close()

print(f"📊 混淆矩阵图片保存于：{conf_matrix_path}")

