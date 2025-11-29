# import os
# from pathlib import Path
# from ultralytics import YOLO
# import pandas as pd
# import shutil
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ===== 配置 =====
# MODEL_PATH = "/cms/user/huangsuyun/runs/detect/train3/weights/best.pt"  # pollutant + wire 的模型
# IMAGE_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new/images/test")
# LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new/labels/test")

# OUTPUT_DIR = Path("runs/detect/afbonding/predict/test_iou5_conf10")
# RESULT_DIR = Path("runs/detect/afbonding/results_iou5_conf10")  # 按分类保存图片
# REPORT_FILE = "misclassified_pollutant_wire_test.csv"

# # 创建输出目录
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# (RESULT_DIR / "pollutant_wire_to_bkg").mkdir(parents=True, exist_ok=True)
# (RESULT_DIR / "bkg_to_pollutant_wire").mkdir(parents=True, exist_ok=True)
# (RESULT_DIR / "pollutant_wire_to_pollutant_wire").mkdir(parents=True, exist_ok=True)

# # 加载模型
# model = YOLO(MODEL_PATH)

# print("开始对 test 数据集进行预测 ...")
# results = model.predict(
#     source=str(IMAGE_DIR),
#     save=True,
#     save_txt=True,
#     project=str(OUTPUT_DIR.parent),
#     name=OUTPUT_DIR.name,
#     exist_ok=True,
#     conf=0.1,
#     iou=0.5
# )

# # 扫描预测标签
# labels_pred_dir = OUTPUT_DIR / "labels"
# y_true, y_pred = [], []

# records = []  # 用于 CSV 保存

# if labels_pred_dir.exists():
#     for label_file in LABEL_DIR.glob("*.txt"):
#         # 真实是否有 pollutant / wire
#         has_obj = label_file.stat().st_size > 0

#         # 找对应图片
#         possible_exts = [".bmp", ".BMP", ".Bmp", ".png", ".PNG"]
#         img_path = None
#         for ext in possible_exts:
#             candidate = IMAGE_DIR / (label_file.stem + ext)
#             if candidate.exists():
#                 img_path = candidate
#                 break

#         if img_path is None:
#             print(f"⚠️ 找不到图片 {label_file.stem}")
#             continue

#         # 是否预测为 pollutant / wire
#         pred_file = labels_pred_dir / label_file.name
#         pred_obj = pred_file.exists() and pred_file.stat().st_size > 0

#         y_true.append(1 if has_obj else 0)
#         y_pred.append(1 if pred_obj else 0)

#         # 分类目录
#         if has_obj and not pred_obj:
#             category = "pollutant_wire_to_bkg"  # 漏检
#         elif not has_obj and pred_obj:
#             category = "bkg_to_pollutant_wire"  # 误检
#         elif has_obj and pred_obj:
#             category = "pollutant_wire_to_pollutant_wire"  # 正检
#         else:
#             continue  # background → background 不保存

#         # 找预测图
#         pred_img_path = OUTPUT_DIR / img_path.name
#         if not pred_img_path.exists():
#             pred_img_path = OUTPUT_DIR / (img_path.stem + ".jpg")

#         if pred_img_path.exists():
#             shutil.copy(pred_img_path, RESULT_DIR / category / pred_img_path.name)
#         else:
#             print(f"⚠️ 未找到预测图 {pred_img_path.name}")

#         records.append({
#             "image": str(img_path),
#             "true_has_obj": has_obj,
#             "pred_has_obj": pred_obj,
#             "category": category
#         })

# else:
#     print(f"⚠️ 预测标签目录 {labels_pred_dir} 不存在")

# # 导出 CSV
# df = pd.DataFrame(records)
# df.to_csv(REPORT_FILE, index=False)
# print(f"✅ 检测完成，记录已保存到 {REPORT_FILE}")

# # ===== 混淆矩阵 =====
# cm = confusion_matrix(y_true, y_pred)
# print("\n混淆矩阵：\n", cm)

# print("\n分类报告：\n",
#       classification_report(
#           y_true, y_pred,
#           target_names=["background", "object"]  # object = pollutant 或 wire
#       ))

# # 保存混淆矩阵
# plt.figure(figsize=(8, 6), dpi=300)
# sns.heatmap(
#     cm, annot=True, fmt="d", cmap="Blues",
#     xticklabels=["background", "object"],
#     yticklabels=["background", "object"],
#     square=True, linewidths=0.5
# )
# plt.xlabel("Predicted", fontsize=14, weight="bold")
# plt.ylabel("True", fontsize=14, weight="bold")
# plt.title("Test Set Confusion Matrix", fontsize=16, weight="bold")
# plt.tight_layout()

# conf_matrix_path = OUTPUT_DIR / "confusion_matrix_highres.png"
# plt.savefig(conf_matrix_path, dpi=300)
# plt.close()

# print(f"📊 混淆矩阵保存于：{conf_matrix_path}")



# ##################################################分成三类
# import os
# from pathlib import Path
# from ultralytics import YOLO
# import pandas as pd
# import shutil
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==========================
# #        配置路径
# # ==========================
# MODEL_PATH = "/cms/user/huangsuyun/runs/detect/train9/weights/best.pt"

# IMAGE_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new1/images/test")
# LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new1/labels/test")

# OUTPUT_DIR = Path("runs/detect/afbonding/predict_train9/test_iou5_conf10")
# RESULT_DIR = Path("runs/detect/afbonding/predict_train9/results_iou5_conf10")  
# REPORT_FILE = "misclassified_pollutant_wire_background.csv"

# CLASSES = ["pollutant", "wire", "background"]  # 0 / 1 / 2

# # ==========================
# # 创建目录
# # ==========================
# for c in [
#     "pollutant_to_wire", "pollutant_to_bkg",
#     "wire_to_pollutant", "wire_to_bkg",
#     "bkg_to_pollutant", "bkg_to_wire",
#     "correct_pollutant", "correct_wire", "correct_bkg",
# ]:
#     (RESULT_DIR / c).mkdir(parents=True, exist_ok=True)

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # ==========================
# #       加载模型
# # ==========================
# model = YOLO(MODEL_PATH)

# print("开始预测 test 数据集 ...")

# results = model.predict(
#     source=str(IMAGE_DIR),
#     save=True,
#     save_txt=True,
#     project=str(OUTPUT_DIR.parent),
#     name=OUTPUT_DIR.name,
#     exist_ok=True,
#     conf=0.1,
#     iou=0.5
# )

# # 预测标签目录
# labels_pred_dir = OUTPUT_DIR / "labels"

# y_true, y_pred = [], []
# records = []

# possible_exts = [".bmp", ".BMP", ".Bmp", ".png", ".PNG"]

# # ==========================
# #        逐张处理
# # ==========================
# for label_file in LABEL_DIR.glob("*.txt"):

#     # ---- 真实标签 ----
#     if label_file.stat().st_size == 0:
#         true_cls = 2  # background
#     else:
#         with open(label_file) as f:
#             ids = [int(x.split()[0]) for x in f.readlines()]
#         if 0 in ids:
#             true_cls = 0
#         elif 1 in ids:
#             true_cls = 1
#         else:
#             true_cls = 2

#     # ---- 找图片 ----
#     img_path = None
#     for ext in possible_exts:
#         p = IMAGE_DIR / (label_file.stem + ext)
#         if p.exists():
#             img_path = p
#             break
#     if img_path is None:
#         print(f"⚠️ 找不到图片：{label_file.stem}")
#         continue

#     # ---- 预测标签 ----
#     pred_file = labels_pred_dir / label_file.name
#     if not pred_file.exists() or pred_file.stat().st_size == 0:
#         pred_cls = 2  # background
#     else:
#         with open(pred_file) as f:
#             pred_ids = [int(x.split()[0]) for x in f.readlines()]
#         if 0 in pred_ids:
#             pred_cls = 0
#         elif 1 in pred_ids:
#             pred_cls = 1
#         else:
#             pred_cls = 2

#     y_true.append(true_cls)
#     y_pred.append(pred_cls)

#     # ==========================
#     #    分类保存预测图
#     # ==========================
#     pred_img = OUTPUT_DIR / img_path.name
#     if not pred_img.exists():
#         pred_img = OUTPUT_DIR / (img_path.stem + ".jpg")

#     if pred_img.exists():

#         # pollutant 类
#         if true_cls == 0:
#             if pred_cls == 0:
#                 tgt = "correct_pollutant"
#             elif pred_cls == 1:
#                 tgt = "pollutant_to_wire"
#             else:
#                 tgt = "pollutant_to_bkg"

#         # wire 类
#         elif true_cls == 1:
#             if pred_cls == 1:
#                 tgt = "correct_wire"
#             elif pred_cls == 0:
#                 tgt = "wire_to_pollutant"
#             else:
#                 tgt = "wire_to_bkg"

#         # background 类
#         else:
#             if pred_cls == 0:
#                 tgt = "bkg_to_pollutant"
#             elif pred_cls == 1:
#                 tgt = "bkg_to_wire"
#             else:
#                 tgt = "correct_bkg"

#         shutil.copy(pred_img, RESULT_DIR / tgt / pred_img.name)

#     records.append({
#         "image": str(img_path),
#         "true": CLASSES[true_cls],
#         "pred": CLASSES[pred_cls],
#     })

# # ==========================
# # 保存记录
# # ==========================
# df = pd.DataFrame(records)
# df.to_csv(REPORT_FILE, index=False)
# print(f"✅ CSV 结果已保存：{REPORT_FILE}")

# # ==========================
# #     3×3 混淆矩阵
# # ==========================
# cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
# print("\n混淆矩阵：\n", cm)

# print("\n分类报告：\n")
# print(classification_report(
#     y_true, y_pred,
#     labels=[0,1,2],
#     target_names=CLASSES
# ))

# # ==========================
# #    绘制混淆矩阵
# # ==========================
# plt.figure(figsize=(8, 6), dpi=300)
# sns.heatmap(
#     cm, annot=True, fmt="d", cmap="Blues",
#     xticklabels=CLASSES,
#     yticklabels=CLASSES,
#     square=True, linewidths=0.5
# )
# plt.xlabel("Predicted", fontsize=14)
# plt.ylabel("True", fontsize=14)
# plt.title("Confusion Matrix (3 Classes)", fontsize=16)
# plt.tight_layout()

# conf_matrix_path = OUTPUT_DIR / "confusion_matrix_3class.png"
# plt.savefig(conf_matrix_path, dpi=300)
# plt.close()

# print(f"📊 混淆矩阵图片保存于：{conf_matrix_path}")

######################################################二分类

################################################## 只分成两类：wire / background
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


MODEL_PATH = "/cms/user/huangsuyun/dataset/runs/afterbonding/train_1000bg_aug_alb/weights/best.pt" 
IMAGE_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg/images/test")
LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg/labels/test")
OUTPUT_DIR = Path("/cms/user/huangsuyun/dataset/runs/detect/afbonding/predict_selected_wire_1000bg/test_iou5_conf5")
RESULT_DIR = Path("/cms/user/huangsuyun/dataset/runs/detect/afbonding/predict_selected_wire_1000bg/results_iou5_conf5")

# MODEL_PATH = "/publicfs/cms/user/huangsuyun/dataset/runs/afterbonding/train_1_105_1500_aug_alb/weights/best.pt"  # 训练好的模型权重
# IMAGE_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new_wire_105_5_1500_aug/images/test")
# LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new_wire_105_5_1500_aug/labels/test")
# OUTPUT_DIR = Path("/cms/user/huangsuyun/dataset/runs/detect/afbonding/predict_wire_105_1500_aug_alb/test_iou5_conf5")
# RESULT_DIR = Path("/cms/user/huangsuyun/dataset/runs/detect/afbonding/predict_wire_105_1500_aug_alb/results_iou5_conf5")  # 存放分类保存的图片
REPORT_FILE = "misclassified_glue_test.csv"


CLASSES = ["wire", "background"]  # 0 / 1

# ==========================
# 创建分类保存目录
# ==========================
for c in [
    "wire_to_bkg", "bkg_to_wire",
    "correct_wire", "correct_bkg"
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
    conf=0.05,
    iou=0.5
)

# 预测标签目录
labels_pred_dir = OUTPUT_DIR / "labels"

y_true, y_pred = [], []
records = []

possible_exts = [".bmp", ".BMP", ".Bmp", ".png", ".PNG", ".jpg", ".JPG"]

# ==========================
#        逐张处理
# ==========================
for label_file in LABEL_DIR.glob("*.txt"):

    # ---- 真实标签 ----
    if label_file.stat().st_size == 0:
        true_cls = 1  # background
    else:
        with open(label_file) as f:
            ids = [int(x.split()[0]) for x in f.readlines()]
        true_cls = 0 if 0 in ids else 1  # 只有 wire / bkg

    # ---- 找原图 ----
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
        pred_cls = 1  # background
    else:
        with open(pred_file) as f:
            ids = [int(x.split()[0]) for x in f.readlines()]
        pred_cls = 0 if 0 in ids else 1

    y_true.append(true_cls)
    y_pred.append(pred_cls)

    # ---- 预测结果图片 ----
    pred_img = OUTPUT_DIR / img_path.name
    if not pred_img.exists():
        pred_img = OUTPUT_DIR / (img_path.stem + ".jpg")

    if pred_img.exists():

        # wire 类
        if true_cls == 0:
            if pred_cls == 0:
                tgt = "correct_wire"
            else:
                tgt = "wire_to_bkg"

        # background 类
        else:
            if pred_cls == 1:
                tgt = "correct_bkg"
            else:
                tgt = "bkg_to_wire"

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
#     混淆矩阵 (2×2)
# ==========================
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
print("\n混淆矩阵：\n", cm)

print("\n分类报告：\n")
print(classification_report(
    y_true, y_pred,
    labels=[0,1],
    target_names=CLASSES
))

# ==========================
#    绘制混淆矩阵
# ==========================
plt.figure(figsize=(6, 5), dpi=300)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    square=True, linewidths=0.5
)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.title("Confusion Matrix (2 Classes)", fontsize=16)
plt.tight_layout()

conf_matrix_path = OUTPUT_DIR / "confusion_matrix_2class.png"
plt.savefig(conf_matrix_path, dpi=300)
plt.close()

print(f"📊 混淆矩阵图片保存于：{conf_matrix_path}")
