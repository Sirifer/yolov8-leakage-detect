import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ===== 输入你的四个数 =====
TP = 109
TN = 930
FP = 28
FN = 11

# ===== 计算指标 =====
def safe_div(a, b): return a / b if b != 0 else 0
total = TP + TN + FP + FN

accuracy = safe_div(TP + TN, total)
precision = safe_div(TP, TP + FP)
recall = safe_div(TP, TP + FN)
f1 = safe_div(2 * precision * recall, precision + recall)
specificity = safe_div(TN, TN + FP)
fpr = safe_div(FP, FP + TN)
fnr = safe_div(FN, FN + TP)
mcc_denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
mcc = (TP * TN - FP * FN) / mcc_denom if mcc_denom != 0 else 0

# ===== 打印结果 =====
print("=== Confusion Matrix ===")
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print("\n=== Metrics ===")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall (TPR) : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"MCC          : {mcc:.4f}")
print(f"FPR={fpr:.4f}, FNR={fnr:.4f}")

# ===== 保存报告到 CSV =====
metrics = {
    "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    "Accuracy": accuracy, "Precision": precision, "Recall": recall,
    "Specificity": specificity, "F1": f1, "MCC": mcc, "FPR": fpr, "FNR": fnr
}
pd.DataFrame([metrics]).to_csv("confusion_report.csv", index=False)
print("\n✅ CSV 已保存为 confusion_report.csv")

# ===== 绘制混淆矩阵（原始 + 归一化） =====
cm = np.array([[TN, FP],
               [FN, TP]])

# 原始矩阵
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=13)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Neg", "Pred Pos"])
ax.set_yticklabels(["Actual Neg", "Actual Pos"])
ax.set_title("Confusion Matrix (Raw)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# 归一化矩阵（按行归一化）
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=13)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Neg", "Pred Pos"])
ax.set_yticklabels(["Actual Neg", "Actual Pos"])
ax.set_title("Confusion Matrix (Normalized)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix_norm.png", dpi=300)
plt.close()

print("✅ 混淆矩阵已保存为 confusion_matrix.png 与 confusion_matrix_norm.png")
