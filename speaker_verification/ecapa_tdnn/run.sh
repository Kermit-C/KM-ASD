#!/bin/bash

### 将本次作业计费到导师课题组，tutor_project改为导师创建的课题组名
#SBATCH --comment=cross_modal

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=vit

### 指定该作业需要多少个节点
### 注意！没有使用多机并行（MPI/NCCL等），下面参数写1！不要多写，多写了也不会加>速程序！
#SBATCH --nodes=1

### 指定该作业需要多少个CPU核心
### 注意！一般根据队列的CPU核心数填写，比如cpu队列64核，这里申请64核，并在你的程序中尽量使用多线程充分利用64核资源！
#SBATCH --ntasks=20

### 指定该作业在哪个队列上执行
### 目前可用的队列有 cpu/fat/titan/tesla
#SBATCH --partition=titan
#SBATCH --gpus=1
### 以上参数用来申请所需资

### 本例使用Anaconda中的Python，先将Python添加到环境变量配置好环境变量
export PATH=/opt/app/anaconda3/bin:$PATH
### 激活一个 Anaconda 环境 tf22
source activate xue_torch

### 执行你的作业
python train.py --cfg configs/vit_stat_vox2.yaml --wandb