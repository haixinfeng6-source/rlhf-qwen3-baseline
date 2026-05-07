#!/bin/bash

# RLHF 多卡训练启动脚本
# 支持 4卡 和 8卡 训练

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4卡训练，8卡改为 0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# RLHF 训练方法: PPO (Proximal Policy Optimization)

# GPU 数量
NUM_GPUS=4  # 根据实际情况修改

# 模型和数据集配置
MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="openbmb/UltraFeedback"
OUTPUT_DIR="./output_rlhf_$(date +%Y%m%d_%H%M%S)"

# 训练超参数
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-5
EPOCHS=1
MAX_LENGTH=512

# LoRA 配置
LORA_R=16
LORA_ALPHA=32

echo "=========================================="
echo "Starting RLHF (PPO) training with ${NUM_GPUS} GPUs"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# 使用 torchrun 启动多卡训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train_rlhf.py \
    --model_name=${MODEL_NAME} \
    --dataset_name=${DATASET_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --learning_rate=${LEARNING_RATE} \
    --num_train_epochs=${EPOCHS} \
    --max_length=${MAX_LENGTH} \
    --use_lora \
    --lora_r=${LORA_R} \
    --lora_alpha=${LORA_ALPHA} \
    --use_8bit \
    --gradient_checkpointing \
    --logging_steps=10 \
    --save_steps=500

echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=========================================="
