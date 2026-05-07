# RLHF 集群训练 - 极简配置指南

## 问题分析

根据错误信息，用户在多GPU集群上遇到以下问题：
1. `torchrun` 启动后所有进程都失败 (`ChildFailedError`)
2. TRL 版本兼容性问题 (用户安装了 TRL 1.3.0)
3. 分布式训练初始化问题

## 解决方案

### 步骤 1: 在登录节点 (CPU) 配置环境

```bash
# 1. 创建干净的虚拟环境
python3.10 -m venv ~/rlhf-cluster
source ~/rlhf-cluster/bin/activate

# 2. 安装固定版本的包 (避免版本冲突)
pip install torch==2.1.0 transformers==4.36.0
pip install trl==0.11.0  # 使用稳定版本，避免 1.x 的 API 变化
pip install datasets==2.14.0 accelerate==0.25.0 peft==0.7.0

# 3. 检查安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import importlib.metadata; print(f'TRL: {importlib.metadata.version(\"trl\")}')"
```

### 步骤 2: 准备离线资源

确保 `offline_assets/` 目录已传输到集群，包含：
- `offline_assets/models/Qwen--Qwen3-8B/` - 模型文件
- `offline_assets/datasets/openbmb--UltraFeedback/` - 数据集
- `offline_assets/wheels/` - Python 包 (可选)

### 步骤 3: 使用修复后的训练脚本

使用我们提供的修复版本：
```bash
# 1. 确保脚本有执行权限
chmod +x run_cluster_offline.sh
chmod +x train_rlhf_cluster_fixed.py

# 2. 测试环境
python train_rlhf_cluster_fixed.py --help
```

### 步骤 4: 提交集群作业

创建 SLURM 作业脚本 `submit_rlhf.slurm`:

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

# 加载模块 (根据集群配置)
module load cuda/11.8
module load python/3.10

# 激活环境
source ~/rlhf-cluster/bin/activate

# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 设置分布式环境
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

# 运行训练
cd /path/to/rlhf-qwen3-baseline

torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_rlhf_cluster_fixed.py \
    --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
    --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
    --output_dir="./output_rlhf_${SLURM_JOB_ID}" \
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --use_lora \
    --gradient_checkpointing \
    --max_samples=100  # 测试用，成功后可以移除
```

### 步骤 5: 提交作业并监控

```bash
# 提交作业
sbatch submit_rlhf.slurm

# 查看作业状态
squeue -u $USER

# 查看输出日志
tail -f rlhf-<job_id>.out
tail -f rlhf-<job_id>.err
```

## 关键修复点

### 1. TRL 导入问题
- 原代码中的复杂版本检查逻辑有问题
- 新脚本使用简化的导入逻辑
- 强制使用 TRL 0.11.0 (稳定版本)

### 2. 分布式训练初始化
- 修复了 `torch.distributed.init_process_group()` 的调用时机
- 添加了正确的环境变量设置
- 改进了错误处理和清理

### 3. 离线模式支持
- 设置 `HF_HUB_OFFLINE=1` 等环境变量
- 直接使用本地路径加载模型和数据集
- 避免任何网络请求

## 测试流程

### 测试 1: 单卡测试 (在登录节点)
```bash
# 测试脚本是否能正常运行
python train_rlhf_cluster_fixed.py \
    --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
    --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
    --output_dir="./test_output" \
    --max_samples=10  # 只测试10个样本
```

### 测试 2: 多卡测试 (在GPU节点)
```bash
# 使用交互式会话测试
srun --gres=gpu:4 --time=1:00:00 --pty bash

# 在交互式会话中运行
bash run_cluster_offline.sh
```

## 常见错误及解决

### 错误 1: `ImportError: cannot import name 'AutoModelForCausalLMWithValueHead'`
**原因**: TRL 版本不兼容
**解决**: 
```bash
pip uninstall trl -y
pip install trl==0.11.0
```

### 错误 2: `ChildFailedError` (所有进程失败)
**原因**: 分布式训练初始化问题
**解决**:
1. 检查 `MASTER_ADDR` 和 `MASTER_PORT` 设置
2. 确保所有节点网络互通
3. 使用修复后的脚本

### 错误 3: CUDA out of memory
**解决**:
1. 减小 `--per_device_train_batch_size` (默认 1)
2. 增加 `--gradient_accumulation_steps` (默认 8)
3. 启用 `--gradient_checkpointing`
4. 使用 `--use_8bit` (如果支持)

### 错误 4: 找不到模型或数据集
**解决**:
1. 检查路径是否正确
2. 确保离线资源已传输
3. 使用绝对路径

## 性能优化建议

### 1. 显存优化
```bash
--per_device_train_batch_size=1
--gradient_accumulation_steps=16  # 增加累积步数
--gradient_checkpointing  # 启用检查点
--use_8bit  # 8-bit量化
```

### 2. 训练速度优化
```bash
--max_length=256  # 减小序列长度
--max_samples=1000  # 限制样本数
--logging_steps=50  # 减少日志频率
```

### 3. 稳定性优化
```bash
--seed=42  # 固定随机种子
--max_grad_norm=1.0  # 梯度裁剪
--warmup_steps=100  # 学习率预热
```

## 完整工作流程

1. **准备阶段** (登录节点)
   - 配置 Python 环境
   - 传输离线资源
   - 测试单卡运行

2. **测试阶段** (GPU节点)
   - 申请交互式会话
   - 测试多卡运行
   - 验证分布式训练

3. **生产阶段** (作业调度)
   - 创建作业脚本
   - 提交长期作业
   - 监控训练进度

4. **验证阶段**
   - 检查输出文件
   - 验证模型保存
   - 评估训练结果

## 紧急恢复

如果训练中途失败：

```bash
# 1. 检查错误日志
cat rlhf-<job_id>.err

# 2. 从检查点恢复
torchrun ... train_rlhf_cluster_fixed.py \
    --resume_from_checkpoint="./output_rlhf/checkpoint-500" \
    ...其他参数

# 3. 调整参数后重新提交
# 减小批次大小、增加累积步数等
```

## 支持

如有问题，按以下顺序检查：
1. 作业脚本的资源请求 (GPU数量、内存、时间)
2. 环境变量设置 (离线模式、分布式设置)
3. 文件路径和权限
4. 包版本兼容性

**关键**: 始终先进行小规模测试，确认无误后再进行完整训练。