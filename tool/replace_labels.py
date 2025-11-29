# import shutil
# from pathlib import Path

# # 原数据目录（旧标签位置）
# root = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall")

# # 修改后的 pollutant / wire 目录
# # src_pollutant = root / "selected/pollutant"
# # src_wire = root / "selected/wire"
# src=path("/cms/user/huangsuyun/dataset/samples/bfbondingall/selected_signal")
# # 总共的源目录
# src_dirs = [src_pollutant, src_wire]

# print("开始替换标签文件...\n")

# for src_dir in src_dirs:
#     if not src_dir.exists():
#         print(f"❌ 路径不存在：{src_dir}")
#         continue

#     for txt_file in src_dir.glob("*.txt"):
#         dst_file = root / f"{txt_file.name}"  # 原始标签就在 root 目录下

#         if dst_file.exists():
#             shutil.copy(txt_file, dst_file)
#             print(f"✔ 已覆盖：{dst_file.name}")
#         else:
#             print(f"⚠ 原目录不存在该文件，跳过：{dst_file.name}")

# print("\n🎉 替换完成！")

import shutil
from pathlib import Path

# 原始标签目录
labels_root = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/labels")

# selected_signal 目录
src_dir = Path("/cms/user/huangsuyun/dataset/samples/bfbondingall/selected_signal")

print("开始替换标签文件...\n")

splits = ["train", "val", "test"]

for txt_file in src_dir.glob("*.txt"):
    filename = txt_file.name
    replaced = False

    # 在 train/val/test 中查找同名文件
    for split in splits:
        dst_file = labels_root / split / filename

        if dst_file.exists():
            shutil.copy(txt_file, dst_file)
            print(f"✔ 覆盖：labels/{split}/{filename}")
            replaced = True
            break

    if not replaced:
        print(f"⚠ 找不到同名标签（跳过）：{filename}")

print("\n🎉 替换完成！")
