# 使用离线资源进行 RLHF 训练

## 概述

您已经下载了所有必要的离线资源到 `offline_assets/` 目录。本指南说明如何直接使用这些资源进行训练，无需网络连接。

## 离线资源结构

```
offline_assets/
├── models/
│   └── Qwen--Qwen3-8B/          # Qwen3-8B 模型
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer.json
├── datasets/
│   └── openbmb--UltraFeedback/  # UltraFeedback 数据集
│       ├── dataset_dict.json
│       └── train/ (数据文件)
└── wheels/                      # Python 依赖包
    ├── torch-*.whl
    ├── transformers-*.whl
    └── ... (其他依赖)
```

## 环境配置（无网络）

### 1. 安装离线 Python 包

```bash
# 进入项目目录
cd rlhf-qwen3-baseline

# 使用离线 wheels 安装依赖
pip install --no-index --find-links=offline_assets/wheels -r requirements.txt
```

### 2. 或者使用离线安装脚本

```bash
# 运行离线安装脚本
bash install_offline.sh
```

## 训练配置

### 使用离线配置文件

编辑或使用 `config_offline.yaml`：

```yaml
model:
  name: "./offline_assets/models/Qwen--Qwen3-8B"
dataset:
  name: "./offline_assets/datasets/openbmb--UltraFeedback"
```

### 运行离线训练

#### 方法 1：使用离线脚本

```bash
# 单卡训练
bash run_offline.sh

# 多卡训练
bash run_offline_multi_gpu.sh
```

#### 方法 2：直接运行

```bash
# 设置离线环境变量
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 运行训练
python train_rlhf.py \
    --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
    --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
    --output_dir="./output_offline" \
    --use_lora \
    --use_8bit
```

## 集群环境配置

### 创建集群作业脚本

创建 `submit_offline_job.slurm`：

```bash
#!/bin/bash
#SBATCH --job-name=rlhf-offline
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 使用绝对路径
MODEL_PATH="/path/to/rlhf-qwen3-baseline/offline_assets/models/Qwen--Qwen3-8B"
DATASET_PATH="/path/to/rlhf-qwen3-baseline/offline_assets/datasets/openbmb--UltraFeedback"

# 运行训练
torchrun --nproc_per_node=4 train_rlhf.py \
    --model_name="${MODEL_PATH}" \
    --dataset_name="${DATASET_PATH}" \
    --output_dir="./output_offline" \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing
```

### 提交作业

```bash
sbatch submit_offline_job.slurm
```

## 验证离线资源

### 检查资源完整性

```bash
# 运行离线资源测试
python test_offline_assets.py

# 检查模型
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./offline_assets/models/Qwen--Qwen3-8B', trust_remote_code=True)
print('✓ Tokenizer 加载成功')
"

# 检查数据集
python -c "
from datasets import load_from_disk
dataset = load_from_disk('./offline_assets/datasets/openbmb--UltraFeedback')
print(f'✓ 数据集加载成功: {len(dataset)} 样本')
"
```

### 测试离线训练

```bash
# 小规模测试
python train_rlhf.py \
    --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
    --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
    --max_samples=10 \
    --output_dir="./test_output" \
    --num_train_epochs=1
```

## 常见问题解决

### 1. 路径问题

**问题**: `FileNotFoundError` 或路径错误
**解决**: 使用绝对路径

```bash
# 获取绝对路径
realpath offline_assets/models/Qwen--Qwen3-8B

# 在脚本中使用
MODEL_PATH=$(realpath offline_assets/models/Qwen--Qwen3-8B)
```

### 2. 权限问题

**问题**: 无法读取文件
**解决**: 检查文件权限

```bash
# 检查权限
ls -la offline_assets/models/Qwen--Qwen3-8B/

# 修改权限（如果需要）
chmod -R 755 offline_assets/
```

### 3. 环境变量问题

**问题**: 仍然尝试下载
**解决**: 确保设置正确的环境变量

```bash
# 强制离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/tmp  # 避免使用默认缓存
```

## 快速开始

### 步骤 1：验证环境

```bash
cd rlhf-qwen3-baseline
python test_offline_assets.py
```

### 步骤 2：安装依赖

```bash
# 如果已经安装，跳过此步
bash install_offline.sh
```

### 步骤 3：测试训练

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python train_rlhf.py \
    --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
    --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
    --max_samples=5 \
    --output_dir="./test_output"
```

### 步骤 4：完整训练

```bash
# 单卡
bash run_offline.sh

# 或多卡
bash run_offline_multi_gpu.sh

# 或集群作业
sbatch submit_offline_job.slurm
```

## 资源管理

### 清理临时文件

```bash
# 清理测试输出
rm -rf test_output/

# 清理 PyTorch 缓存
rm -rf ~/.cache/torch
rm -rf ~/.cache/huggingface
```

### 备份重要结果

```bash
# 备份训练结果
tar -czf output_backup_$(date +%Y%m%d).tar.gz output_offline/

# 备份配置
cp config_offline.yaml config_backup_$(date +%Y%m%d).yaml
```

## 注意事项

1. **路径一致性**: 确保所有脚本使用相同的路径
2. **环境隔离**: 在虚拟环境中操作
3. **资源监控**: 监控 GPU 显存使用
4. **日志记录**: 保存完整的训练日志

## 支持

如果遇到问题：

1. 检查 `offline_assets/` 目录结构
2. 验证文件权限
3. 检查环境变量设置
4. 查看训练日志中的具体错误

使用离线资源可以完全避免网络问题，特别适合无网络集群环境。