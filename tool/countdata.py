#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# ===== 配置路径 =====
ROOT = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg_noaug/labels")
splits = ["train", "val", "test"]

summary = []

for split in splits:
    folder = ROOT / split
    if not folder.exists():
        continue

    all_labels = list(folder.glob("*.txt"))
    total = len(all_labels)
    empty = 0

    for f in all_labels:
        if f.stat().st_size == 0:  # 空文件
            empty += 1

    signal = total - empty  # 有内容 = 信号
    summary.append({
        "subset": split,
        "total_labels": total,
        "signal": signal,
        "background": empty
    })

# 输出统计结果
df = pd.DataFrame(summary)
print("\n=== 标签扫描结果 ===")
print(df.to_string(index=False))
print("\n总计：")
print(df[["total_labels", "signal", "background"]].sum())

# #!/usr/bin/env python3
# from pathlib import Path
# import pandas as pd

# # ===== 配置 =====
# ROOT = Path("/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/new/labels")
# splits = ["train", "val", "test"]

# summary = []
# all_classes = set()   # 收集所有类别编号

# # 第一次扫描：统计所有类别
# for split in splits:
#     folder = ROOT / split
#     if not folder.exists():
#         continue

#     for txt in folder.glob("*.txt"):
#         if txt.stat().st_size == 0:
#             continue
#         try:
#             for line in txt.read_text().strip().splitlines():
#                 cls = int(line.split()[0])
#                 all_classes.add(cls)
#         except:
#             print(f"⚠️ 跳过异常文件: {txt}")

# # 将类别排序固定
# all_classes = sorted(list(all_classes))
# print(f"检测到类别: {all_classes}")

# # 第二次扫描：统计每个 split 的数量
# for split in splits:
#     folder = ROOT / split
#     if not folder.exists():
#         continue

#     all_labels = list(folder.glob("*.txt"))
#     total = len(all_labels)
#     empty = 0

#     class_count = {f"class_{cls}": 0 for cls in all_classes}

#     for f in all_labels:
#         if f.stat().st_size == 0:
#             empty += 1
#             continue

#         try:
#             lines = f.read_text().strip().splitlines()
#             for line in lines:
#                 cls = int(line.split()[0])
#                 class_count[f"class_{cls}"] += 1
#         except:
#             print(f"⚠️ 文件解析失败: {f}")

#     signal = total - empty

#     entry = {
#         "subset": split,
#         "total_labels": total,
#         "background": empty,
#         "signal": signal
#     }
#     entry.update(class_count)
#     summary.append(entry)

# # ---- 输出结果 ----
# df = pd.DataFrame(summary).fillna(0)

# # 只对数值列转 int
# numeric_cols = df.columns.drop("subset")
# df[numeric_cols] = df[numeric_cols].astype(int)

# print("\n=== 标签扫描结果（按类别） ===")
# print(df.to_string(index=False))

# total_df = df[numeric_cols].sum()

# print("\n=== 总计 ===")
# print(total_df)
