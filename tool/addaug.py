import os
import random
from pathlib import Path
import shutil

# ================== 路径 ==================
aug_root = Path("//cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/af_a")  # 增强后信号
dataset_root = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new2/new1")

# 目标路径
img_dirs = {
    "train": dataset_root / "images/train",
    "val": dataset_root / "images/val",
    "test": dataset_root / "images/test",
}

label_dirs = {
    "train": dataset_root / "labels/train",
    "val": dataset_root / "labels/val",
    "test": dataset_root / "labels/test",
}

# ================== 读取增强后的信号 ==================
aug_images = sorted(list(aug_root.glob("*.bmp")) +
                    list(aug_root.glob("*.BMP")) +
                    list(aug_root.glob("*.png"))+
                    list(aug_root.glob("*.jpg")))

aug_data = []
for img in aug_images:
    label = aug_root / f"{img.stem}.txt"
    if label.exists() and label.stat().st_size > 0:
        aug_data.append((img, label))

print(f"增强信号数量: {len(aug_data)}")

# ================== 统计原始 new1 的比例 ==================
def count_files(path):
    return len(list(path.glob("*.bmp"))) + len(list(path.glob("*.png"))) + len(list(path.glob("*.BMP")))

n_train = count_files(img_dirs["train"])
n_val = count_files(img_dirs["val"])
n_test = count_files(img_dirs["test"])

total = n_train + n_val + n_test
p_train = n_train / total
p_val = n_val / total
p_test = n_test / total

print(f"原始比例: train={p_train:.3f}, val={p_val:.3f}, test={p_test:.3f}")

# ================== 按比例分配增强信号 ==================
random.shuffle(aug_data)

train_n = int(len(aug_data) * p_train)
val_n = int(len(aug_data) * p_val)
test_n = len(aug_data) - train_n - val_n

train_set = aug_data[:train_n]
val_set = aug_data[train_n:train_n + val_n]
test_set = aug_data[train_n + val_n:]

splits = {
    "train": train_set,
    "val": val_set,
    "test": test_set
}

# ================== 拷贝增强后的信号 ==================
for split, data_list in splits.items():
    for img, lbl in data_list:
        dst_img = img_dirs[split] / img.name
        dst_lbl = label_dirs[split] / lbl.name

        shutil.copy(img, dst_img)
        shutil.copy(lbl, dst_lbl)

print("🎉 增强信号成功按比例加入 new1 数据集！")
