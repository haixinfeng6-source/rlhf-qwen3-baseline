#!/bin/bash

# 单卡测试脚本
# 用于测试 RLHF 训练是否能在单卡上正常工作

echo "=========================================="
echo "单卡 RLHF 测试"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 测试参数
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="openbmb/UltraFeedback"
OUTPUT_DIR="./test_output_$(date +%Y%m%d_%H%M%S)"
MAX_SAMPLES=10  # 只测试10个样本

echo "模型: ${MODEL_NAME}"
echo "数据集: ${DATASET_NAME}"
echo "输出目录: ${OUTPUT_DIR}"
echo "最大样本数: ${MAX_SAMPLES}"
echo "=========================================="

# 检查环境
echo "检查环境..."
python -c "
import sys
print(f'Python版本: {sys.version}')

import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

# 检查 TRL 导入
try:
    from trl.models import AutoModelForCausalLMWithValueHead
    print('✓ TRL 0.11.0+ 导入方式正常')
except ImportError:
    try:
        from trl import AutoModelForCausalLMWithValueHead
        print('✓ TRL 0.10.0 导入方式正常')
    except ImportError as e:
        print(f'✗ TRL 导入失败: {e}')
        sys.exit(1)
"

# 运行测试训练
echo "开始测试训练..."
python train_rlhf.py \
    --model_name=${MODEL_NAME} \
    --dataset_name=${DATASET_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --max_samples=${MAX_SAMPLES} \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=1 \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing \
    --logging_steps=1 \
    --save_steps=5

echo "=========================================="
echo "测试完成!"
echo "如果单卡测试成功，再尝试多卡训练: bash run_multi_gpu.sh"
echo "=========================================="