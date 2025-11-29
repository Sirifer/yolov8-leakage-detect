# import cv2
# import albumentations as A
# from pathlib import Path
# import shutil

# # ---------- 配置 ----------
# IMG_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/af_wire")
# LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/af_wire")
# DST_IMG_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/af_wire_aug")
# DST_LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/af_wire_aug")

# DST_IMG_DIR.mkdir(exist_ok=True)
# DST_LABEL_DIR.mkdir(exist_ok=True)

# # ---------- 支持的图片格式 ----------
# IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".BMP"]

# # ---------- 定义增强 ----------
# # transform = A.Compose([
# #     A.HorizontalFlip(p=1),
# #     A.VerticalFlip(p=1),
# #     A.Rotate(limit=30, p=1),
# #     A.RandomScale(scale_limit=0.2, p=0.7),
# # ])


# transform = A.Compose([
#     # ---------- 几何增强 ----------
#     A.HorizontalFlip(p=0.5),                      # 随机水平翻转
#     A.VerticalFlip(p=0.5),                        # 随机垂直翻转
#     A.RandomRotate90(p=0.5),                      # 随机 0/90/180/270 度旋转
#     # A.Rotate(limit=30, p=0.8),                    # 随机 ±30° 旋转
#     # A.RandomScale(scale_limit=0.2, p=0.8),        # 随机缩放 ±20%
#     A.ShiftScaleRotate(
#         shift_limit=0.05, 
#         # scale_limit=0.1, 
#         rotate_limit=15, 
#         p=0.8
#     ),                                            # 平移/缩放/旋转综合增强

#     # ---------- 颜色增强 ----------
#     A.ColorJitter(
#         brightness=0.3,       # 亮度变化 ±30%
#         contrast=0.3,         # 对比度 ±30%
#         saturation=0.3,       # 饱和度 ±30%
#         hue=0.1,              # 色相变化 ±10%
#         p=0.7
#     ),

#     A.RandomBrightnessContrast(
#         brightness_limit=0.2, 
#         contrast_limit=0.2, 
#         p=0.5
#     ),                         # 亮度对比度调整

#     # A.RGBShift(
#     #     r_shift_limit=10,     # R/G/B 通道随机偏移
#     #     g_shift_limit=10,
#     #     b_shift_limit=10,
#     #     p=0.5
#     # ),

#     # A.HueSaturationValue(
#     #     hue_shift_limit=10, 
#     #     sat_shift_limit=15,
#     #     val_shift_limit=10,
#     #     p=0.5
#     # ),                         # HSV 色彩变化，适合自然图像增强

#     # # ---------- 噪声与模糊 ----------
#     # A.GaussNoise(var_limit=(5, 20), p=0.4),       # 高斯噪声
#     # A.MotionBlur(blur_limit=3, p=0.3),            # 运动模糊
#     # A.GaussianBlur(blur_limit=(3, 5), p=0.3),     # 高斯模糊

#     # # ---------- 直方图类增强 ----------
#     # A.CLAHE(clip_limit=4, p=0.3),                 # 局部直方图均衡化（增强细节）

#     # # ---------- Gamma ----------
#     # A.RandomGamma(gamma_limit=(70, 130), p=0.4),  # Gamma 变化

#     # ToTensorV2()
# ])



# # ---------- 遍历所有标签文件 ----------
# label_files = list(LABEL_DIR.glob("*.txt"))
# print(f"共检测到标签文件: {len(label_files)} 个\n开始增强...")

# for label_file in label_files:
#     # 自动匹配图片文件
#     img_path = None
#     for ext in IMG_EXTS:
#         test_path = IMG_DIR / f"{label_file.stem}{ext}"
#         if test_path.exists():
#             img_path = test_path
#             break

#     if img_path is None:
#         print(f"[跳过] 找不到图片: {label_file.stem}")
#         continue

#     img = cv2.imread(str(img_path))
#     if img is None:
#         print(f"[错误] 图片读取失败: {img_path}")
#         continue

#     print(f"增强: {img_path.name}")

#     # 每张图增强 5 次
#     for i in range(5):
#         augmented = transform(image=img)['image']
#         dst_img = DST_IMG_DIR / f"{label_file.stem}_aug{i}.jpg"
#         cv2.imwrite(str(dst_img), augmented)

#         # 标签复制
#         dst_label = DST_LABEL_DIR / f"{label_file.stem}_aug{i}.txt"
#         shutil.copy(label_file, dst_label)

# print("\n✅ 数据增强完成！")
# print(f"输出图片目录：{DST_IMG_DIR}")
# print(f"输出标签目录：{DST_LABEL_DIR}")
# 




#--------------------------八种————————————————-----------------------------
# import cv2
# from pathlib import Path
# import albumentations as A

# IMG_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg/images/test")
# LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg/labels/test")
# DST_IMG_DIR = IMG_DIR
# DST_LABEL_DIR = LABEL_DIR

# DST_IMG_DIR.mkdir(exist_ok=True)
# DST_LABEL_DIR.mkdir(exist_ok=True)

# IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".BMP"]

# label_files = [f for f in LABEL_DIR.glob("*.txt") if f.stat().st_size > 0]

# print(f"共检测到标签文件: {len(label_files)} 个\n开始增强...")

# # 定义8种变换组合
# transform_list = [
#     A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),   # H
#     A.Compose([A.VerticalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),     # V
#     A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])), # H+V
#     A.Compose([A.Rotate(limit=[90,90], p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])), # 90°
#     A.Compose([A.Rotate(limit=[90,90], p=1.0), A.HorizontalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
#     A.Compose([A.Rotate(limit=[90,90], p=1.0), A.VerticalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
#     A.Compose([A.Rotate(limit=[90,90], p=1.0), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
# ]

# for label_file in label_files:

#     # 找到对应图片
#     img_path = None
#     for ext in IMG_EXTS:
#         test_path = IMG_DIR / f"{label_file.stem}{ext}"
#         if test_path.exists():
#             img_path = test_path
#             break
#     if img_path is None:
#         print(f"[跳过] 找不到图片: {label_file.stem}")
#         continue

#     img = cv2.imread(str(img_path))
#     if img is None:
#         print(f"[错误] 图片读取失败: {img_path}")
#         continue

#     # 读取 YOLO 标签
#     bboxes = []
#     labels = []
#     with open(label_file, "r") as f:
#         for line in f:
#             data = line.strip().split()
#             if len(data) != 5:
#                 continue
#             c, x, y, w, h = map(float, data)
#             labels.append(int(c))
#             bboxes.append([x, y, w, h])

#     if len(bboxes) == 0:
#         print(f"[跳过背景图] {label_file.name} 没有 bbox")
#         continue

#     # 应用8种变换
#     for i, transform in enumerate(transform_list):
#         augmented = transform(image=img, bboxes=bboxes, class_labels=labels)
#         aug_img = augmented["image"]
#         aug_bboxes = augmented["bboxes"]
#         aug_labels = augmented["class_labels"]

#         dst_img = DST_IMG_DIR / f"{label_file.stem}_aug{i}.jpg"
#         cv2.imwrite(str(dst_img), aug_img)

#         dst_label = DST_LABEL_DIR / f"{label_file.stem}_aug{i}.txt"
#         with open(dst_label, "w") as f:
#             for cls, (x, y, w, h) in zip(aug_labels, aug_bboxes):
#                 f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# print("\n✅ 数据增强完成！")

import cv2
from pathlib import Path

IMG_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg/images/val")
LABEL_DIR = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg/labels/val")
DST_IMG_DIR = IMG_DIR
DST_LABEL_DIR = LABEL_DIR

DST_IMG_DIR.mkdir(exist_ok=True)
DST_LABEL_DIR.mkdir(exist_ok=True)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".BMP"]

label_files = [f for f in LABEL_DIR.glob("*.txt") if f.stat().st_size > 0]

def yolo_flip(bboxes, flip_horizontal=False, flip_vertical=False):
    new_bboxes = []
    for x, y, w, h in bboxes:
        if flip_horizontal:
            x = 1 - x
        if flip_vertical:
            y = 1 - y
        new_bboxes.append([x, y, w, h])
    return new_bboxes

def yolo_rotate90(bboxes):
    new_bboxes = []
    for x, y, w, h in bboxes:
        new_x = y
        new_y = 1 - x
        new_w = h
        new_h = w
        new_bboxes.append([new_x, new_y, new_w, new_h])
    return new_bboxes

for label_file in label_files:

    # 找到对应图片
    img_path = None
    for ext in IMG_EXTS:
        test_path = IMG_DIR / f"{label_file.stem}{ext}"
        if test_path.exists():
            img_path = test_path
            break
    if img_path is None:
        print(f"[跳过] 找不到图片: {label_file.stem}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[错误] 图片读取失败: {img_path}")
        continue

    # 读取 YOLO 标签
    labels = []
    bboxes = []
    with open(label_file, "r") as f:
        for line in f:
            data = line.strip().split()
            if len(data) != 5:
                continue
            c, x, y, w, h = map(float, data)
            labels.append(int(c))
            bboxes.append([x, y, w, h])

    if len(bboxes) == 0:
        print(f"[跳过背景图] {label_file.name} 没有 bbox")
        continue

    aug_idx = 0
    h, w = img.shape[:2]

    # --- 原图翻转组合 ---
    for flip_h, flip_v in [(1,0), (0,1), (1,1)]:  # H, V, H+V
        aug_img = img.copy()
        aug_bboxes = yolo_flip(bboxes, flip_horizontal=flip_h, flip_vertical=flip_v)

        if flip_h:
            aug_img = cv2.flip(aug_img, 1)
        if flip_v:
            aug_img = cv2.flip(aug_img, 0)

        dst_img = DST_IMG_DIR / f"{label_file.stem}_aug{aug_idx}.jpg"
        cv2.imwrite(str(dst_img), aug_img)

        dst_label = DST_LABEL_DIR / f"{label_file.stem}_aug{aug_idx}.txt"
        with open(dst_label, "w") as f:
            for cls, (bx, by, bw, bh) in zip(labels, aug_bboxes):
                f.write(f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
        aug_idx += 1

    # --- 旋转90° + 翻转组合（包括不翻转） ---
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_bboxes = yolo_rotate90(bboxes)

    for flip_h, flip_v in [(0,0), (1,0), (0,1), (1,1)]:  # 不翻转, H, V, H+V
        aug_img = rotated_img.copy()
        aug_bboxes = yolo_flip(rotated_bboxes, flip_horizontal=flip_h, flip_vertical=flip_v)

        if flip_h:
            aug_img = cv2.flip(aug_img, 1)
        if flip_v:
            aug_img = cv2.flip(aug_img, 0)

        dst_img = DST_IMG_DIR / f"{label_file.stem}_aug{aug_idx}.jpg"
        cv2.imwrite(str(dst_img), aug_img)

        dst_label = DST_LABEL_DIR / f"{label_file.stem}_aug{aug_idx}.txt"
        with open(dst_label, "w") as f:
            for cls, (bx, by, bw, bh) in zip(labels, aug_bboxes):
                f.write(f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
        aug_idx += 1

print("\n✅ 数据增强完成！")
