import os
from pathlib import Path
from fpdf import FPDF
from ultralytics import YOLO
import pandas as pd
from PyPDF2 import PdfMerger

# ========== 自定义路径 ==========
ROOT_DIR = Path("/cms/user/huangsuyun/yolov8/2025_9_15")  # 多个模块文件夹的上级目录
FINAL_PDF = Path("/cms/user/huangsuyun/yolov8/reports/all_leakage2.pdf")  # 最终合并PDF输出路径
MODEL_PATH = "/cms/user/huangsuyun/yolov8/runs/detect/train/weights/best.pt"

CELL_PHOTO_MAP = {  # cell号映射表
       36 : "1",
    38 : "2",
    40 : "30",
    42 : "13",
    44 : "24",
    46 : "34",
    48 : "5",
    50 : "25",
    52 : "82",
    54 : "22",
    56 : "70",
    58 : "83",
    60 : "98",
    62 : "58",
    64 : "48",
    66 : "141",
    68 : "61",
    70 : "60",
    72 : "142",
    74 : "129",
    76 : "130",
    78 : "156",
    80 : "180",
    82 : "171",
    84 : "136",
    86 : "94",
    88 : "138",
    90 : "163",
    92 : "190",
    94 : "192",
    96 : "126",
    98 : "176",
    100 : "154",
    102 : "177",
    104 : "3",
    106 : "4",
    108 : "7",
    110 : "27",
    112 : "28",
    114 : "51",
    116 : "63",
    118 : "74",
    120 : "104",
    122 : "105",
    124 : "91",
    126 : "77",
    128 : "93",
    130 : "64",
    132 : "80",
    134 : "111",
    136 : "140",
    138 : "139",
    140 : "168",
    142 : "153",
    144 : "179",
    146 : "189",
    148 : "150",
    150 : "149",
    152 : "174",
    154 : "161",
    156 : "172",
    158 : "184",
    160 : "196",
    162 : "186",
    164 : "198",
    166 : "169",
    168 : "132",
    170 : "133",
    172 : "120",
    174 : "112",
    176 : "99",
    178 : "116",
    180 : "102",
    182 : "86",
    184 : "118",
    186 : "85",
    188 : "71",
    190 : "87",
    192 : "57",
    194 : "41",
    196 : "31",
    198 : "corner_9",
    200 : "corner_18",
    202 : "corner_95",
    204 : "corner_197",
    206 : "corner_191",
    208 : "corner_81",
    210 : "66",
    212 : "52",
    214 : "67",
    216 : "54",
    218 : "55",
    220 : "47",
    222 : "8",
    224 : "124",
    226 : "122",
    228 : "185",
    230 : "81",
}

# ========== 创建 PDF ==========
def create_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    try:
        # 使用中文字体
        pdf.add_font("SourceHanSansSC", style="", fname="SourceHanSansSC-Regular.otf", uni=True)
        pdf.add_font("SourceHanSansSC", style="B", fname="SourceHanSansSC-Bold.otf", uni=True)
        pdf.set_font("SourceHanSansSC", "B", 16)
    except:
        # 如果字体文件不存在，则用默认Arial
        pdf.set_font("Arial", "B", 16)
    return pdf

def add_images_to_pdf(pdf, image_paths, module_id):
    pdf.add_page()
    pdf.set_font("SourceHanSansSC" if "SourceHanSansSC" in pdf.fonts else "Arial", "B", 14)
    pdf.cell(0, 10, f"Leakage Detection Report - {module_id}", ln=True, align="C")

    # ---- 布局参数 ----
    margin = 10
    cols = 5
    width = (210 - 2 * margin - (cols - 1) * 2) / cols  # 每张宽度
    row_height = width * 0.75 + 10  # 图片 + 下方文字空间
    y = 25

    for idx, img_path in enumerate(image_paths):
        x = margin + (idx % cols) * (width + 2)
        if idx % cols == 0 and idx != 0:
            y += row_height
            if y > 260:  # 超出页面则新开页
                pdf.add_page()
                y = 25

        # 从文件名提取编号
        try:
            num = int(Path(img_path).stem.split("-")[-2])
            cell = CELL_PHOTO_MAP.get(num, "?")
        except:
            cell = "?"

        # 插入图片
        pdf.image(img_path, x=x, y=y, w=width)
        pdf.set_font("SourceHanSansSC" if "SourceHanSansSC" in pdf.fonts else "Arial", "", 8)
        pdf.text(x + 2, y + width * 0.75 + 5, f"cell {cell}")

# ========== YOLO 检测 ==========
def detect_glue_leakage(module_path):
    model = YOLO(MODEL_PATH)
    result = model.predict(source=module_path, conf=0.25, save=True, save_txt=True)
    pred_dir = result[0].save_dir  # YOLO 输出目录

    records = []
    for r in result:
        for box in r.boxes:
            cls_name = r.names[int(box.cls[0])]
            if cls_name == "glue":
                records.append({
                    "image": r.path,
                    "class": cls_name,
                    "confidence": float(box.conf[0])
                })

    df = pd.DataFrame(records)
    if not df.empty:
        csv_path = Path(module_path) / "leakage_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"🧾 漏胶信息已保存: {csv_path}")
    return df, pred_dir

# ========== 每个模块生成单独 PDF ==========
def generate_pdf_for_module(module_path):
    module_id = Path(module_path).name.split("_before_")[0]
    print(f"\n🔍 处理模块 {module_id} ...")
    df, pred_dir = detect_glue_leakage(module_path)
    if df is None or df.empty:
        print(f"✅ 未检测到漏胶，跳过 {module_id}")
        return None

    glue_images = sorted(set(Path(row["image"]).stem for _, row in df.iterrows()))
    rendered_imgs = [str(p) for p in Path(pred_dir).glob("*.jpg") if Path(p).stem in glue_images]
    if not rendered_imgs:
        print("⚠️ 未找到 glue 图片")
        return None

    pdf = create_pdf()
    add_images_to_pdf(pdf, rendered_imgs, module_id)
    output = Path(module_path) / f"{module_id}_leakage_report.pdf"
    pdf.output(str(output))
    print(f"📄 模块报告生成完成: {output}")
    return output

# ========== 主流程 ==========
def process_all_modules(root_dir, final_pdf_path):
    pdf_files = []
    for folder in sorted(Path(root_dir).iterdir()):
        if folder.is_dir():
            pdf_path = generate_pdf_for_module(folder)
            if pdf_path:
                pdf_files.append(pdf_path)

    if not pdf_files:
        print("⚠️ 没有检测到任何模块报告，不生成总PDF")
        return

    merger = PdfMerger()
    for pdf in pdf_files:
        merger.append(str(pdf))
    final_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    merger.write(str(final_pdf_path))
    merger.close()
    print(f"\n✅ 所有模块合并完成: {final_pdf_path}")

# ========== 执行 ==========
if __name__ == "__main__":
    process_all_modules(ROOT_DIR, FINAL_PDF)
