#!/bin/bash

# 离线多卡训练启动脚本 (Linux)

echo "=========================================="
echo "RLHF 离线多卡训练"
echo "=========================================="

# 检查资源是否存在
if [ ! -d "offline_assets/models/Qwen--Qwen3-8B" ]; then
    echo "[错误] 模型文件不存在"
    echo "请确保 offline_assets 目录已正确传输"
    exit 1
fi

if [ ! -d "offline_assets/datasets/openbmb--UltraFeedback" ]; then
    echo "[错误] 数据集文件不存在"
    echo "请确保 offline_assets 目录已正确传输"
    exit 1
fi

echo "[检查] 资源文件完整"

# 设置训练参数
MODEL_PATH="./offline_assets/models/Qwen--Qwen3-8B"
DATASET_PATH="./offline_assets/datasets/openbmb--UltraFeedback"
OUTPUT_DIR="./output_offline"

# GPU 配置
NUM_GPUS=4
MASTER_PORT=29500

echo ""
echo "训练配置:"
echo "  模型: $MODEL_PATH"
echo "  数据集: $DATASET_PATH"
echo "  输出: $OUTPUT_DIR"
echo "  GPU 数量: $NUM_GPUS"
echo ""

# 检查 GPU
echo "检查 GPU 状态:"
python -c "import torch; print(f'  可用 GPU: {torch.cuda.device_count()}')"
echo ""

# 询问是否继续
read -p "是否开始训练? (y/n): " CONTINUE
if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始多卡训练..."
echo "=========================================="

# 设置可见 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 使用 torchrun 启动多卡训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_rlhf.py \
    --model_name=$MODEL_PATH \
    --dataset_name=$DATASET_PATH \
    --output_dir=$OUTPUT_DIR \
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --use_lora \
    --lora_r=16 \
    --lora_alpha=32 \
    --use_8bit \
    --gradient_checkpointing \
    --logging_steps=10 \
    --save_steps=500 \
    --max_length=512

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: $OUTPUT_DIR"
echo ""
