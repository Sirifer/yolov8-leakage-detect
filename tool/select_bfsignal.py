import os
import shutil
from pathlib import Path

# ======= 配置路径 =======
root = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall")

images_root = root / "images"
labels_root = root / "labels"

# 输出目录
out_dir = root / "selected_signal"
out_dir.mkdir(parents=True, exist_ok=True)

# 图片格式
img_exts = [".bmp", ".BMP", ".png", ".PNG", ".jpg", ".jpeg"]

count_signal = 0

print("开始扫描所有图像...")

for split in ["train", "val", "test"]:
    img_dir = images_root / split
    label_dir = labels_root / split

    for img_path in img_dir.rglob("*"):
        if img_path.suffix not in img_exts:
            continue

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # 标签为空 → 不是 signal
        if label_path.stat().st_size == 0:
            continue

        # 复制到输出目录
        dst_img = out_dir / f"{img_path.name}"
        dst_label = out_dir / f"{label_path.name}"

        shutil.copy(img_path, dst_img)
        shutil.copy(label_path, dst_label)

        count_signal += 1

print("\n🎉 完成筛选！")
print(f"信号类图片数量: {count_signal}")
print(f"输出目录: {out_dir}")
