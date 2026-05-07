# RLHF 训练 - 集群环境配置指南

## 概述

本指南介绍如何在 HPC/GPU 集群上配置和运行 RLHF 训练。典型的工作流程是：

1. **登录节点 (CPU)**: 配置环境、下载资源
2. **计算节点 (GPU)**: 运行训练任务

## 环境配置

### 1. 在登录节点配置环境

```bash
# 创建虚拟环境（推荐）
python3.10 -m venv ~/rlhf-env
source ~/rlhf-env/bin/activate

# 或者使用 conda
conda create -n rlhf python=3.10 -y
conda activate rlhf
```

### 2. 安装依赖

```bash
# 安装核心依赖
pip install torch==2.1.0 transformers==4.36.0 trl==0.11.0

# 安装其他必要包
pip install datasets==2.14.0 accelerate==0.25.0 peft==0.7.0

# 安装工具包
pip install rich tensorboard
```

### 3. 下载离线资源（如果集群无网络）

```bash
# 下载模型和数据集
python download_assets.py

# 下载 Python 包
pip download -r requirements.txt -d offline_assets/wheels
```

## 集群作业提交

### SLURM 作业脚本示例

创建 `submit_job.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=rlhf-qwen3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=rlhf-%j.out
#SBATCH --error=rlhf-%j.err

# 加载模块（根据集群配置）
module load cuda/11.8
module load python/3.10

# 激活环境
source ~/rlhf-env/bin/activate

# 设置环境变量
export HF_HOME=/path/to/huggingface/cache
export TRANSFORMERS_CACHE=/path/to/transformers/cache
export HF_DATASETS_CACHE=/path/to/datasets/cache

# 运行训练
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_rlhf.py \
    --model_name=Qwen/Qwen3-8B \
    --dataset_name=openbmb/UltraFeedback \
    --output_dir=./output_rlhf \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --num_train_epochs=1 \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing
```

### 提交作业

```bash
sbatch submit_job.slurm
```

## 离线模式配置

如果集群无法访问互联网，使用离线模式：

### 1. 修改配置文件

编辑 `config_offline.yaml`:

```yaml
model_name: "offline_assets/models/Qwen--Qwen3-8B"
dataset_name: "offline_assets/datasets/openbmb--UltraFeedback"
use_offline: true
```

### 2. 使用离线安装脚本

```bash
# 在登录节点准备
bash install_offline.sh

# 在作业脚本中添加离线设置
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## 多 GPU 训练

### 单节点多卡

```bash
# 4卡训练
bash run_multi_gpu.sh

# 或直接使用 torchrun
torchrun --nproc_per_node=4 train_rlhf.py [参数]
```

### 多节点训练

```bash
# 需要配置 NCCL 和网络
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node1:29500 \
    train_rlhf.py [参数]
```

## 常见问题解决

### 1. CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvidia-smi
python -c "import torch; print(torch.version.cuda)"

# 安装匹配的 PyTorch
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 2. 内存不足

- 启用梯度检查点: `--gradient_checkpointing`
- 使用 8-bit 量化: `--use_8bit`
- 减小批次大小: `--per_device_train_batch_size=1`
- 增加梯度累积: `--gradient_accumulation_steps=16`

### 3. 导入错误

```bash
# 检查 Python 版本
python --version  # 推荐 3.10

# 检查包版本
pip list | grep -E "(torch|transformers|trl|peft)"

# 重新安装特定版本
pip install trl==0.11.0 --force-reinstall
```

## 监控和调试

### 查看作业状态

```bash
# SLURM 命令
squeue -u $USER
sacct -j <job_id>
scontrol show job <job_id>

# 查看输出
tail -f rlhf-<job_id>.out
tail -f rlhf-<job_id>.err
```

### 性能监控

```bash
# GPU 使用情况
nvidia-smi
watch -n 1 nvidia-smi

# 内存使用
htop
free -h
```

### TensorBoard 监控

```bash
# 在登录节点启动 TensorBoard
tensorboard --logdir=./output_rlhf/logs --port=6006 --bind_all

# 通过 SSH 隧道访问
ssh -L 6006:localhost:6006 user@login-node
```

## 最佳实践

### 1. 环境管理
- 为每个项目创建独立环境
- 使用固定版本避免冲突
- 记录所有依赖版本

### 2. 资源管理
- 预估所需资源（GPU、内存、时间）
- 使用合适的队列和优先级
- 设置合理的超时时间

### 3. 数据管理
- 使用共享存储保存模型和数据
- 定期清理临时文件
- 备份重要结果

### 4. 代码管理
- 使用版本控制（Git）
- 记录实验配置
- 保存完整的训练日志

## 快速开始

### 步骤 1: 环境准备
```bash
git clone https://github.com/your-repo/rlhf-qwen3-baseline
cd rlhf-qwen3-baseline
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 步骤 2: 测试环境
```bash
python train_rlhf_cluster.py
```

### 步骤 3: 提交作业
```bash
# 编辑 submit_job.slurm
# 修改资源需求和参数

sbatch submit_job.slurm
```

### 步骤 4: 监控训练
```bash
# 查看作业状态
squeue -u $USER

# 查看训练日志
tail -f rlhf-<job_id>.out
```

## 支持

如有问题，请检查：
1. 作业脚本中的资源请求
2. 环境变量设置
3. 文件路径和权限
4. 集群的特定配置

参考文档：
- [PyTorch 分布式训练](https://pytorch.org/docs/stable/distributed.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [TRL 文档](https://huggingface.co/docs/trl)
- 集群管理员文档