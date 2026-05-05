#!/bin/bash

# 使用 Accelerate 启动多卡训练
# Accelerate 提供更灵活的分布式训练配置

# 配置文件路径
ACCELERATE_CONFIG="accelerate_config.yaml"

# 模型和数据集
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME="openbmb/UltraFeedback"
OUTPUT_DIR="./output_rlhf_accelerate_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Starting RLHF (PPO) training with Accelerate"
echo "Config: ${ACCELERATE_CONFIG}"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "${ACCELERATE_CONFIG}" ]; then
    echo "Creating default accelerate config..."
    accelerate config default
fi

# 启动训练
accelerate launch \
    --config_file=${ACCELERATE_CONFIG} \
    train_rlhf.py \
    --model_name=${MODEL_NAME} \
    --dataset_name=${DATASET_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --num_train_epochs=1 \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing

echo "Training completed! Model saved to ${OUTPUT_DIR}"
