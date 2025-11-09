
本项目用于使用 **YOLOv8 模型** 对图像进行自动化检测，并生成包含检测结果的 PDF 报告。

### 下载并安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
最好放在内存大目录下

安装完成后，**重新登录或执行以下命令**以启用 conda：

```bash
source ~/.bashrc
```


### 创建 YOLOv8 环境

```bash
conda create -n yolov8 python=3.10
conda activate yolov8

```

### 安装 PyTorch 与 ultralytics

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```


### 安装包

生成 PDF 报告所需：

```bash
pip install fpdf PyPDF2
pip install fpdf
```

---

### 运行脚本



运行命令：

```bash
conda activate yolov8
python /<your path>/dataset/predict_photo_pdf.py
```

### 输出结果说明

默认输出路径（可在脚本中修改）：

```
/<your path>/dataset/reports/
```

生成的文件示例：

```
all_leakage_reports.pdf
```
