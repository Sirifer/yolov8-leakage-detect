import os
from pathlib import Path

# 修改为你自己的路径:为没有标签的图片添加标签
image_dir = Path("/cms/user/huangsuyun/dataset/dataset_name/images/train")
label_dir = Path("/cms/user/huangsuyun/dataset/dataset_name/labels/train")

# 确保标签文件夹存在
label_dir.mkdir(parents=True, exist_ok=True)

# 遍历图片
for img_path in image_dir.glob("*.*"):
    stem = img_path.stem
    txt_path = label_dir / f"{stem}.txt"
    if not txt_path.exists():
        txt_path.touch()  # 创建空文件
        print(f"✅ 创建空标签文件: {txt_path}")
