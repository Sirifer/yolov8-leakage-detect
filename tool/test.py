import shutil
from pathlib import Path

# === 路径 ===
SRC_DIR = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/selected_signal")
DST_DIR = Path("/cms/user/huangsuyun/dataset/2025_9_15/MLF3WCIH0209_before_bonding_front")

print("🚀 开始筛选 pollutant（类别 = 1）图片并替换目标目录...\n")

# ========== 1. 收集 pollutant 图片 ==========
pollutant_imgs = []

for txt_file in SRC_DIR.glob("*.txt"):
    if txt_file.stat().st_size == 0:
        continue

    with open(txt_file) as f:
        cls_ids = [int(x.split()[0]) for x in f.readlines()]

    # 只要出现类别 1 → pollutant
    if 1 in cls_ids:
        img_path_bmp = txt_file.with_suffix(".bmp")
        if img_path_bmp.exists():
            pollutant_imgs.append(img_path_bmp)

print(f"找到 pollutant 图片数量: {len(pollutant_imgs)}")

if len(pollutant_imgs) == 0:
    print("❌ 没有找到 pollutant 图片，程序终止。")
    exit()

# ========== 2. 获取目标目录中的图片 ==========
dst_images = sorted(list(DST_DIR.glob("*.BMP")))

print(f"目标目录图片数量: {len(dst_images)}")

if len(dst_images) == 0:
    print("❌ 目标目录没有 bmp 图片")
    exit()

# ========== 3. 开始按顺序替换 ==========
replace_count = 0

for dst_img, src_img in zip(dst_images, pollutant_imgs):
    shutil.copy(src_img, dst_img)
    print(f"✔ 替换 {dst_img.name}  ←  {src_img.name}")
    replace_count += 1

print(f"\n🎉 完成！总共替换了 {replace_count} 张图片。")
print("⚠️ 如果目标目录图片数量 > pollutant 数量，则剩下的不会被替换。")
print("⚠️ 如果 pollutant 图片更多，多余的也不会处理。")
