参考网站：https://docs.ultralytics.com/zh/

## 数据准备

![[Pasted image 20251128143232.png]]
###  收集数据
- 编号
```
/cms/user/huangsuyun/dataset/tool/move.py
```
- 用labelImg标记（我是在mac本地）

```bash
brew install python@3.9
python3.9 -m venv ~/labelimg-env
source ~/labelimg-env/bin/activate
pip install --upgrade pip
pip install pyqt5 lxml labelImg
labelImg
```
 安装成功
```
 (labelimg-env) huangsuyun@huangsuyundeMacBook-Air ~ % pip show labelImg



Name: labelImg

Version: 1.8.6

Summary: LabelImg is a graphical image annotation tool and label object bounding boxes in images

Home-page: https://github.com/tzutalin/labelImg

Author: TzuTa Lin

Author-email: tzu.ta.lin@gmail.com

License: MIT license

Location: /Users/huangsuyun/labelimg-env/lib/python3.9/site-packages

Requires: lxml, pyqt5

Required-by: 

(labelimg-env) huangsuyun@huangsuyundeMacBook-Air ~ %
```

###  划分数据集
sklern train_test_split
```
/cms/user/huangsuyun/dataset/tool/split_data.py
```

数据主要分为 **图像（images）** 和 **标签（labels）** 两类,可以看sample_test/：

```
dataset/
 ├─ images/
 │   ├─ train/
 │   ├─ val/
 │   └─ test/
 └─ labels/
     ├─ train/
     ├─ val/
     └─ test/
```

### 数据增强 (Albumentations)
用了 Albumentations 做数据增强：
- 旋转、翻转、模糊、亮度/对比度调整等常见变换。
-  **YOLO 标签坐标也需要跟随变换**。
- 这个代码只对已经划分数据的train/val/test/里面的“信号”图片zengqiang
- 原图+增强的7张（水平翻转，垂直翻转，水平翻转+垂直翻转，90度+水平翻转，90度+垂直翻转，90度+水平翻转+垂直翻转）
```
/cms/user/huangsuyun/dataset/tool/Augmentation.py
```

---


## YOLOv8 模型训练
### 数据路径
```
/cms/user/huangsuyun/dataset/data.yaml
```
长这样
```yaml

train: /cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg_noaug/images/train

val: /cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg_noaug/images/val

test: /cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/selected_wire_1000bg_noaug/images/test

nc: 1#表示多少类

names: ['wire']

# names: ['pollutant', 'wire']

# names: ['glue', 'pollutant']
```
### 训练
```
/cms/user/huangsuyun/dataset/train_yolov8.py
```
长这样
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(

data='data.yaml',

epochs=100,

imgsz=640,

batch=16,

device=[0,1],

lr0=0.01,

iou=0.5,

conf=0.05,

project='runs/afterbonding', # 自定义主目录

name='train_1000bg_noaug', # 自定义子目录名

exist_ok=True,

)
```
超参数设置：https://docs.ultralytics.com/zh/modes/train/#train-settings
- slurm脚本
```shell
#! /bin/bash

  

######## Part 1: 资源申请 #########

#SBATCH --partition=gpu # 使用 GPU 分区

#SBATCH --qos=cmsnormal

#SBATCH --account=cmsgpu # 账户 (按你集群要求)

#SBATCH --job-name=yolo_glue_train # 作业名

#SBATCH --ntasks=1 # 任务数 (只跑 1 个训练任务)

#SBATCH --cpus-per-task=8 # 每任务 CPU 核数

#SBATCH --mem-per-cpu=4096 # 每 CPU 内存 (MB)，8*4096=32G

#SBATCH --gpus=v100:2 # 申请 1 张 V100

#SBATCH -t 2:00:00 # 最长运行时间

#SBATCH -o /cms/user/huangsuyun/dataset/runs/slurm-%j.out # 输出日志文件

# 加载环境
source /cvmfs/lhcbdev.cern.ch/conda/miniconda/linux-64/prod/etc/profile.d/conda.sh
conda activate /cms/user/huangsuyun/conda/envs/YOLOv8
  
# 配置 YOLOv8 的用户目录，避免权限报错
export YOLO_CONFIG_DIR=/cms/user/huangsuyun/ultralytics_cfg
export ULTRALYTICS_CACHE_DIR=/cms/user/huangsuyun/ultralytics_cache
mkdir -p "$YOLO_CONFIG_DIR" "$ULTRALYTICS_CACHE_DIR"
export MPLCONFIGDIR=/cms/user/huangsuyun/matplotlib_cache
mkdir -p $MPLCONFIGDIR

srun python /cms/user/huangsuyun/dataset/train_yolov8.py
```

训练后的模型一般在（也可以指定路径）
```
runs/
└─ detect/
   └─ train/
	  ├─ weights/
		 └─ best.pt
      ├─ results.png
      ├─ ...
```
比如run_test/
### 模型预测
```python
from ultralytics import YOLO

model = YOLO(MODEL_PATH)
results = model.predict(source="images/test")
results.show()
```
- 预测代码
可生成混淆矩阵，预测的带框的图片比如：
```
/cms/user/huangsuyun/dataset/predict_afbonding.py
```
or（验证）：
```
/cms/user/huangsuyun/dataset/yolo_test.py
```
- 直接使用预测的模型生成pdf
```
/cms/user/huangsuyun/dataset/predict_photo_pdf.py
```

一些用到的工具脚本在
```
/cms/user/huangsuyun/dataset/tool/
```
