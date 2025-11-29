import os

import shutil
from pathlib import Path
root = Path("/publicfs/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new_5")

label_dirs = [root/"labels/train", root/"labels/test",root/"labels/val"]
# label_dirs = ["/cms/user/huangsuyun/dataset/samples/bfbondingall/selected_signal"]

for label_dir in label_dirs:
    for fname in os.listdir(label_dir):
        if fname.endswith(".txt"):
            path = os.path.join(label_dir, fname)
            with open(path, "r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    # 如果类别是15，改为0，否则保持原样（如果有其他类别请调整这里）
                    if cls_id == 1:
                        parts[0] = '0'
                    new_lines.append(" ".join(parts) + "\n")
            with open(path, "w") as f:
                f.writelines(new_lines)
print("类别ID替换完成。")

