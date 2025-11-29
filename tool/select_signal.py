import os
import shutil
from pathlib import Path

# ======= 配置路径 =======
root = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall")

# 输出目录
out_dir = root / "selected"
(out_dir / "pollutant").mkdir(parents=True, exist_ok=True)
(out_dir / "wire").mkdir(parents=True, exist_ok=True)

# 支持的图片格式
img_exts = [".bmp", ".BMP", ".png", ".PNG"]

count_pollutant = 0
count_wire = 0

print("开始扫描所有图像...")

# 遍历所有图片
for img_path in root.rglob("*"):
    # ---- 跳过 selected 目录，避免重复扫描 ----
    if "selected" in img_path.parts:
        continue

    if img_path.suffix not in img_exts:
        continue
    
    # 找对应标签
    label_path = img_path.with_suffix(".txt")
    if not label_path.exists():
        continue

    # 标签内容为空 → background，忽略
    if label_path.stat().st_size == 0:
        continue

    # 读取标签
    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    classes = {int(line.split()[0]) for line in lines}

    # 复制 pollutant
    if 0 in classes:
        dst_img = out_dir / "pollutant" / img_path.name
        dst_label = out_dir / "pollutant" / label_path.name
        if not dst_img.exists():   # 避免 SameFileError
            shutil.copy(img_path, dst_img)
        if not dst_label.exists():
            shutil.copy(label_path, dst_label)
        count_pollutant += 1

    # 复制 wire
    if 1 in classes:
        dst_img = out_dir / "wire" / img_path.name
        dst_label = out_dir / "wire" / label_path.name
        if not dst_img.exists():
            shutil.copy(img_path, dst_img)
        if not dst_label.exists():
            shutil.copy(label_path, dst_label)
        count_wire += 1

print("\n🎉 完成分类！")
print(f"pollutant 数量: {count_pollutant}")
print(f"wire 数量: {count_wire}")
print(f"输出目录: {out_dir}")
