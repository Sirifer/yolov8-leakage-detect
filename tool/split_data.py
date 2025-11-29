# import os
# from pathlib import Path
# import shutil
# from sklearn.model_selection import train_test_split

# # === 1️⃣ 修改你的路径 ===
# root = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall")  # 改成你自己的文件夹路径

# # 创建目标文件夹结构
# for folder in ["images/train", "images/val", "images/test",
#                 "labels/train", "labels/val", "labels/test"]:
#     (root / folder).mkdir(parents=True, exist_ok=True)

# # === 2️⃣ 收集所有图片 ===
# # all_images = sorted([p for p in root.glob("*.bmp")])
# all_images = sorted([
#     p for p in root.rglob("*.[bB][mM][pP]")   # BMP
# ] + [
#     p for p in root.rglob("*.[pP][nN][gG]")   # PNG
# ])

# print(f"✅ 找到 {len(all_images)} 张图片。")



# # === 3️⃣ 用 sklearn 划分训练/验证/测试 ===
# train_imgs, temp_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
# val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

# splits = {
#     "train": train_imgs,
#     "val": val_imgs,
#     "test": test_imgs
# }

# # === 4️⃣ 移动图片 + 标签 ===
# for split, images in splits.items():
#     for img_path in images:
#         # 图片目标路径
#         dst_img = root / f"images/{split}" / img_path.name

#         # 对应的标签路径
#         label_name = img_path.stem + ".txt"
#         src_label = root / label_name
#         dst_label = root / f"labels/{split}" / label_name

#         # 拷贝图片
#         shutil.copy(img_path, dst_img)

#         # 如果有对应 txt 文件就拷贝，否则创建空 txt
#         if src_label.exists():
#             shutil.copy(src_label, dst_label)
#         else:
#             dst_label.touch()

# print("✅ 数据划分完成，YOLOv8 格式已生成！")




# import os
# import random
# from pathlib import Path
# import shutil
# from sklearn.model_selection import train_test_split

# root = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall")
# root_new = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new")

# # 1. 创建 root_new 目录结构
# for folder in ["images/train", "images/val", "images/test",
#                "labels/train", "labels/val", "labels/test"]:
#     (root_new / folder).mkdir(parents=True, exist_ok=True)

# # 2. 只找 root 目录下的图片（不会找子目录）
# all_images = sorted(list(root.glob("*.bmp")) + list(root.glob("*.png")))

# positives = []  # 有标签的图片
# negatives = []  # 空标签（背景）

# for img in all_images:
#     label = root / f"{img.stem}.txt"

#     # 标签必须在 root 下，且非空
#     if label.exists() and label.stat().st_size > 0:
#         positives.append(img)
#     else:
#         negatives.append(img)

# print(f"正样本(有目标): {len(positives)} 张")
# print(f"背景图(空标签): {len(negatives)} 张")

# # 3. 限制背景图数量
# max_background = 800
# negatives = random.sample(negatives, min(max_background, len(negatives)))
# print(f"实际使用背景图: {len(negatives)} 张")

# # 4. 合并
# final_images = positives + negatives

# # 5. 划分
# train_imgs, temp_imgs = train_test_split(final_images, test_size=0.2, random_state=42)
# val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

# splits = {
#     "train": train_imgs,
#     "val": val_imgs,
#     "test": test_imgs
# }

# # 6. 复制到 root_new 结构
# for split, images in splits.items():
#     for img_path in images:

#         # 目标图片路径
#         dst_img = root_new / f"images/{split}" / img_path.name
        
#         # 原标签路径（仍然只在 root 下查找）
#         label_path = root / f"{img_path.stem}.txt"
        
#         # 目标标签路径
#         dst_label = root_new / f"labels/{split}" / f"{img_path.stem}.txt"

#         # 复制图片
#         shutil.copy(img_path, dst_img)

#         # 复制标签（若不存在则创建空文件）
#         if label_path.exists():
#             shutil.copy(label_path, dst_label)
#         else:
#             dst_label.touch()

# print("🎉 数据划分完成！所有输出已保存到 new 文件夹。")


# export OPENBLAS_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1


import os
import random
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
random.seed(42)

# ============= 路径配置 =============
signal_roots = [
    Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire")
    # Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/af_wire_aug")
]
background_root = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/rawdata/afterbondingall")
root_new = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg_noaug")

# ============= 创建输出目录结构 =============
for folder in ["images/train", "images/val", "images/test",
               "labels/train", "labels/val", "labels/test"]:
    (root_new / folder).mkdir(parents=True, exist_ok=True)

# ============= 读取信号图片 =============
positives = []
signal_names = set()

for signal_root in signal_roots:
    signal_images = sorted(
        list(signal_root.glob("*.bmp")) +
        list(signal_root.glob("*.BMP")) +
        list(signal_root.glob("*.jpg")) +
        list(signal_root.glob("*.png"))
    )

    for img in signal_images:
        label = signal_root / f"{img.stem}.txt"
        if label.exists() and label.stat().st_size > 0:  # 仅保留有内容的标签
            positives.append((img, label))
            signal_names.add(img.name)

print(f"信号图片数量（有标签）: {len(positives)}")

# ============= 读取背景图片（没有 txt 文件的） =============
background_images = sorted(
    list(background_root.glob("*.bmp")) +
    list(background_root.glob("*.BMP")) +
    list(background_root.glob("*.png")) +
    list(background_root.glob("*.jpg"))
)

negatives = []

for img in background_images:

    # 跳过 new/selected 目录
    if any(x in str(img.parent) for x in ["new", "new1", "new2", "new3", "new4", "selected"]):
        continue

    # 跳过跟 signal 重名的图片
    if img.name in signal_names:
        continue

    # “无标签文件” → 判定为背景
    label_file = img.with_suffix(".txt")
    if not label_file.exists():
        negatives.append(img)

print(f"背景图片数量（无 txt）: {len(negatives)}")

# 限制背景图数量
max_background = 1000
negatives = random.sample(negatives, min(max_background, len(negatives)))
print(f"实际使用背景图数量: {len(negatives)}")

# ============= 合并数据 =============
final_list = positives.copy()
final_list += [(img, None) for img in negatives]  # 背景图标签为 None

# ============= 划分 train/val/test =============
train_list, temp_list = train_test_split(final_list, test_size=0.2, random_state=42)
val_list, test_list = train_test_split(temp_list, test_size=0.5, random_state=42)

splits = {"train": train_list, "val": val_list, "test": test_list}

# ============= 复制图片和标签 =============
for split, file_list in splits.items():
    for img, lbl in file_list:
        dst_img = root_new / f"images/{split}" / img.name
        dst_lbl = root_new / f"labels/{split}" / f"{img.stem}.txt"

        shutil.copy(img, dst_img)

        if lbl is not None:  # 信号图
            shutil.copy(lbl, dst_lbl)
        else:               # 背景图 → 创建空标签
            dst_lbl.touch()

print("\n🎉 数据划分完成！输出在 new4 文件夹。")
