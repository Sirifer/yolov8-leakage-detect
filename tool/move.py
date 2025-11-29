#!/usr/bin/env python3
import os
import shutil
import csv
import re
from pathlib import Path

# ==== 自定义配置 ====
ROOT_DIR = Path("/cms/user/huangsuyun/dataset/samples/beforebonding")
DST_DIR = ROOT_DIR / "all"
KEYWORD = "*"      # 只复制名字里包含这个关键字的文件夹
EXT = ".bmp"            # 只复制这种类型的图片（忽略大小写）

# ==== 辅助函数 ====
FIG_RE = re.compile(r"^fig0*([0-9]+)(?:(_\d+)?)\.[a-zA-Z0-9]+$")

def find_max_index(dst_dir):
    """找出 all 文件夹中现有 fig 文件的最大编号"""
    max_n = 0
    if not dst_dir.exists():
        return 0
    for f in dst_dir.iterdir():
        m = FIG_RE.match(f.name)
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except:
                pass
    return max_n

def gather_images(folder):
    """在 folder 下递归收集 .bmp 文件"""
    imgs = []
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix.lower() == EXT.lower():
            imgs.append(p)
    return imgs

# ==== 主流程 ====
def main():
    DST_DIR.mkdir(exist_ok=True)
    start_idx = find_max_index(DST_DIR) + 1
    mapping_file = DST_DIR / "mapping.csv"
    write_header = not mapping_file.exists()

    # 找出所有包含 "before" 的文件夹
    folders = [f for f in ROOT_DIR.iterdir() if f.is_dir() and KEYWORD in f.name]
    print(f"找到 {len(folders)} 个文件夹匹配 '{KEYWORD}'")

    count = 0
    with open(mapping_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["original_path", "new_name"])
        idx = start_idx
        for folder in folders:
            imgs = gather_images(folder)
            print(f"  - {folder.name}: {len(imgs)} 张图片")
            for img in imgs:
                new_name = f"fig{idx}.bmp"
                dst_path = DST_DIR / new_name
                while dst_path.exists():  # 确保不覆盖
                    idx += 1
                    new_name = f"fig{idx}.bmp"
                    dst_path = DST_DIR / new_name
                shutil.copy2(img, dst_path)
                writer.writerow([str(img), new_name])
                idx += 1
                count += 1
    print(f"\n✅ 复制完成：共 {count} 张图片复制到 {DST_DIR}")
    print(f"📄 映射表已更新：{mapping_file}")

if __name__ == "__main__":
    main()
