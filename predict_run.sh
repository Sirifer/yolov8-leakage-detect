#! /bin/bash

######## Part 1: 资源申请 #########
#SBATCH --partition=gpu            
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu          
#SBATCH --job-name=yolo_glue_predict  
#SBATCH --ntasks=1                   # 任务数
#SBATCH --cpus-per-task=8            # 每任务 CPU 核数
#SBATCH --mem-per-cpu=4096           # 每 CPU 内存 (MB)
#SBATCH --gpus=v100:1                # 申请 1 张 V100
#SBATCH -t 00:15:00                  
#SBATCH -o /cms/user/huangsuyun/dataset/runs/slurm-predict-%j.out   

######## Part 2: 环境设置 #########
source /cvmfs/lhcbdev.cern.ch/conda/miniconda/linux-64/prod/etc/profile.d/conda.sh
conda activate /cms/user/huangsuyun/conda/envs/YOLOv8

######## Part 3: 启动预测 #########
srun python /cms/user/huangsuyun/dataset/predict_photo.py
