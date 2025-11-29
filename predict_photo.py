# import os, sys, csv, glob
# import cv2
# from ultralytics import YOLO

# # ==== 配置 ====
# MODEL_PATH = "runs/detect/train9/weights/best.pt"  # 你的模型
# IMG_DIR    = "/cms/user/huangsuyun/dataset/samples/beforebonding/F4CQH00558_before_bonding_front"  # 待检测图片文件夹
# OUT_DIR    = "/cms/user/huangsuyun/dataset/samples/beforebonding/annotated"              # 只保存带框图
# CONF       = 0.5
# IOU        = 0.45
# # ============

# os.makedirs(OUT_DIR, exist_ok=True)

# # 载入模型与类别映射
# model = YOLO(MODEL_PATH)
# name2id = {v: k for k, v in model.names.items()}
# if "glue" not in name2id:
#     print("模型中未找到名为 'glue' 的类别。现有类别：", list(model.names.values()))
#     sys.exit(1)
# glue_id = name2id["glue"]

# # 收集图片
# exts = ("*.jpg","*.jpeg","*.png","*.BMP","*.tif","*.tiff","*.webp")
# img_list = []
# for e in exts:
#     img_list.extend(glob.glob(os.path.join(IMG_DIR, e)))
# img_list.sort()
# if not img_list:
#     print("未在目录中找到图片：", IMG_DIR); sys.exit(1)

# # 结果清单
# csv_path = os.path.join(os.path.dirname(OUT_DIR), "glue_results.csv")
# total = len(img_list)
# has_glue_cnt = 0

# with open(csv_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["image","num_glue","conf_list"])

#     # 逐张推理
#     for r in model.predict(source=img_list, conf=CONF, iou=IOU, stream=True, verbose=True):
#         boxes = r.boxes
#         glue_confs = []
#         if boxes is not None and len(boxes) > 0:
#             for b in boxes:
#                 if int(b.cls.item()) == glue_id:
#                     glue_confs.append(float(b.conf.item()))

#         if glue_confs:
#             has_glue_cnt += 1
#             # 只保存带框图
#             annotated = r.plot()
#             save_path = os.path.join(OUT_DIR, os.path.basename(r.path))
#             cv2.imwrite(save_path, annotated)

#             # 写CSV（可选；便于留痕）
#             writer.writerow([os.path.basename(r.path), len(glue_confs),
#                              ";".join(f"{c:.3f}" for c in glue_confs)])

# print(f"完成：共处理 {total} 张图片，其中检测到漏胶的图片 {has_glue_cnt} 张。")
# print(f"- 带框结果保存到：{OUT_DIR}")
# print(f"- 结果清单CSV：{csv_path}")

# import os
# from pathlib import Path
# from ultralytics import YOLO
# import pandas as pd
# import shutil

# # ===== 配置 =====
# MODEL_PATH = "runs/detect/train/weights/best.pt"
# IMAGE_DIR = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/images/test")
# LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/labels/test")
# OUTPUT_DIR = Path("runs/detect/predict/test")
# MISCLASS_DIR = Path("runs/detect/misclassified_images")
# REPORT_FILE = "misclassified_glue_test.csv"

# # 创建输出目录
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# MISCLASS_DIR.mkdir(parents=True, exist_ok=True)

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
#     conf=0.25
# )

# # 扫描预测 txt 文件，找出被误判的 glue
# labels_pred_dir = OUTPUT_DIR / "labels"
# misclassified_images = []

# if labels_pred_dir.exists():
#     for label_file in LABEL_DIR.glob("*.txt"):
#         # 仅统计原本有 glue 的图片
#         if label_file.stat().st_size > 0:
#             # 支持大小写匹配 .bmp / .BMP
#             possible_exts = [".bmp", ".BMP"]
#             img_path = None
#             for ext in possible_exts:
#                 candidate = IMAGE_DIR / (label_file.stem + ext)  # ✅ 修正
#                 if candidate.exists():
#                     img_path = candidate
#                     break


#             if img_path is None:
#                 print(f"警告：找不到对应图片 {label_file.stem}，跳过")
#                 continue

#             pred_file = labels_pred_dir / label_file.name
#             if not pred_file.exists() or pred_file.stat().st_size == 0:
#                 misclassified_images.append(str(img_path))
#                 shutil.copy(img_path, MISCLASS_DIR / img_path.name)
# else:
#     print(f"警告：预测标签目录 {labels_pred_dir} 不存在")

# # 输出 CSV
# df = pd.DataFrame({"misclassified_glue": misclassified_images})
# df.to_csv(REPORT_FILE, index=False)

# print(f"完成！共有 {len(misclassified_images)} 张 glue 信号被误判为背景")
# print(f"列表已保存到 {REPORT_FILE}")
# print(f"误判图片已保存到 {MISCLASS_DIR}")
import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# # ===== 配置 =====
# MODEL_PATH = "runs/detect/train/weights/best.pt"  # 训练好的模型权重
# IMAGE_DIR = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/images/test")
# LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/labels/test")
# OUTPUT_DIR = Path("runs/detect/predict/test1")
# MISCLASS_DIR = Path("runs/detect/misclassified_images1")  # 存放误判图片
# REPORT_FILE = "misclassified_glue_test.csv"

# # 创建输出目录
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# MISCLASS_DIR.mkdir(parents=True, exist_ok=True)

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
#     conf=0.1
# )

# # 扫描预测 txt 文件，找出被误判的 glue
# labels_pred_dir = OUTPUT_DIR / "labels"
# misclassified_images = []

# y_true = []
# y_pred = []

# if labels_pred_dir.exists():
#     for label_file in LABEL_DIR.glob("*.txt"):
#         has_glue = label_file.stat().st_size > 0
#         # 支持大小写匹配 .bmp / .BMP / .Bmp
#         possible_exts = [".bmp", ".BMP", ".Bmp"]
#         img_path = None
#         for ext in possible_exts:
#             candidate = IMAGE_DIR / (label_file.stem + ext)
#             if candidate.exists():
#                 img_path = candidate
#                 break

#         if img_path is None:
#             print(f"⚠️ 警告：找不到对应图片 {label_file.stem}，跳过")
#             continue

#         pred_file = labels_pred_dir / label_file.name
#         pred_glue = pred_file.exists() and pred_file.stat().st_size > 0

#         # 收集混淆矩阵数据
#         y_true.append(1 if has_glue else 0)
#         y_pred.append(1 if pred_glue else 0)

#         # 收集误判图片（真实有 glue 但是预测为空）
#         if has_glue and not pred_glue:
#             misclassified_images.append(str(img_path))
#             shutil.copy(img_path, MISCLASS_DIR / img_path.name)
# else:
#     print(f"⚠️ 警告：预测标签目录 {labels_pred_dir} 不存在")

# # 输出 CSV
# df = pd.DataFrame({"misclassified_glue": misclassified_images})
# df.to_csv(REPORT_FILE, index=False)

# print(f"完成！共有 {len(misclassified_images)} 张 glue 信号被误判为背景")
# print(f"列表已保存到 {REPORT_FILE}")
# print(f"误判图片已保存到 {MISCLASS_DIR}")

# # ===== 生成混淆矩阵和指标 =====
# cm = confusion_matrix(y_true, y_pred)
# print("混淆矩阵：\n", cm)
# print("\n分类报告：\n", classification_report(y_true, y_pred, target_names=["background","glue"]))

# # 保存混淆矩阵图片
# plt.figure(figsize=(6,5), dpi=300)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["background","glue"], yticklabels=["background","glue"])
# plt.xlabel("predicted")
# plt.ylabel("true")
# plt.title("Test Set Confusion Matrix")
# plt.tight_layout()
# conf_matrix_path = OUTPUT_DIR / "confusion_matrix.png"
# plt.savefig(conf_matrix_path)
# plt.close()
# print(f"混淆矩阵图片已保存到 {conf_matrix_path}")

import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 配置 =====
MODEL_PATH = "/cms/user/huangsuyun/dataset/runs/afterbonding/train_1_105_1000_aug_alb/weights/best.pt"  # 训练好的模型权重
IMAGE_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new_wire_105_5_1000_aug/images/test")
LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new_wire_105_5_1000_aug/labels/test")
OUTPUT_DIR = Path("/cms/user/huangsuyun/dataset/runs/detect/afbonding/predict_wire_105_1000_aug_alb/test_iou5_conf10_alb")
RESULT_DIR = Path("/cms/user/huangsuyun/dataset/runs/detect/afbonding/predict_wire_105_1000_aug_alb/results_iou5_conf10_alb")  # 存放分类保存的图片
REPORT_FILE = "misclassified_glue_test.csv"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(RESULT_DIR / "glue_to_bkg").mkdir(parents=True, exist_ok=True)
(RESULT_DIR / "bkg_to_glue").mkdir(parents=True, exist_ok=True)
(RESULT_DIR / "glue_to_glue").mkdir(parents=True, exist_ok=True)

# 加载模型
model = YOLO(MODEL_PATH)

print("开始对 test 数据集进行预测 ...")
results = model.predict(
    source=str(IMAGE_DIR),
    save=True,  # 不保存所有图片
    save_txt=True,
    project=str(OUTPUT_DIR.parent),
    name=OUTPUT_DIR.name,
    exist_ok=True,
    conf=0.05,
    iou=0.5
)

# 扫描预测 txt 文件
labels_pred_dir = OUTPUT_DIR / "labels"
y_true, y_pred = [], []

records = []  # 保存详细结果记录（用于CSV输出）

if labels_pred_dir.exists():
    for label_file in LABEL_DIR.glob("*.txt"):
        has_glue = label_file.stat().st_size > 0  # 真实是否有glue
        possible_exts = [".bmp", ".BMP", ".Bmp"]
        img_path = None
        for ext in possible_exts:
            candidate = IMAGE_DIR / (label_file.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"⚠️ 警告：找不到对应图片 {label_file.stem}，跳过")
            continue

        pred_file = labels_pred_dir / label_file.name
        pred_glue = pred_file.exists() and pred_file.stat().st_size > 0  # 是否预测为glue

        y_true.append(1 if has_glue else 0)
        y_pred.append(1 if pred_glue else 0)

        # 分类存放图片
        if has_glue and not pred_glue:
            category = "glue_to_bkg"
        elif not has_glue and pred_glue:
            category = "bkg_to_glue"
        elif has_glue and pred_glue:
            category = "glue_to_glue"
        else:
            continue  # background → background 不保存

        # 找预测后带框的图片（YOLO输出在 predict/test1 下）
        pred_img_path = OUTPUT_DIR / img_path.name
        if not pred_img_path.exists():
            # 有时 YOLO 会改成 .jpg 输出
            pred_img_path = OUTPUT_DIR / (img_path.stem + ".jpg")
        if pred_img_path.exists():
            shutil.copy(pred_img_path, RESULT_DIR / category / pred_img_path.name)
        else:
            print(f"⚠️ 未找到预测图 {pred_img_path.name}，仅跳过保存")

        records.append({"image": str(img_path), "true": has_glue, "pred": pred_glue, "category": category})
else:
    print(f"⚠️ 警告：预测标签目录 {labels_pred_dir} 不存在")

# 输出 CSV
df = pd.DataFrame(records)
df.to_csv(REPORT_FILE, index=False)
print(f"✅ 检测完成，共处理 {len(df)} 张图片。详细记录已保存到 {REPORT_FILE}")
print(f"分类图片路径：{RESULT_DIR}")
print(f" - glue_to_bkg：漏检的 glue\n - bkg_to_glue：误检的背景\n - glue_to_glue：正确检出的 glue")

# ===== 生成混淆矩阵和分类报告 =====
cm = confusion_matrix(y_true, y_pred)
print("\n混淆矩阵：\n", cm)
print("\n分类报告：\n", classification_report(y_true, y_pred, target_names=["background", "glue"]))

# 保存高清混淆矩阵图
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["background", "signal"],
    yticklabels=["background", "signal"],
    square=True,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 14, "weight": "bold"}
)
plt.xlabel("Predicted", fontsize=14, weight="bold")
plt.ylabel("True", fontsize=14, weight="bold")
plt.title("Test Set Confusion Matrix", fontsize=16, weight="bold", pad=15)
plt.xticks(fontsize=12, weight="bold")
plt.yticks(fontsize=12, weight="bold", rotation=0)
plt.tight_layout()
conf_matrix_path = OUTPUT_DIR / "confusion_matrix_highres.png"
plt.savefig(conf_matrix_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"📊 高清混淆矩阵已保存到 {conf_matrix_path}")
