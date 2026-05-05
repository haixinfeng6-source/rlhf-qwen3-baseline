#!/bin/bash

# 离线训练启动脚本 (Linux/Mac)

echo "=========================================="
echo "RLHF 离线训练"
echo "=========================================="

# 检查资源是否存在
if [ ! -d "offline_assets/models/Qwen--Qwen2.5-7B-Instruct" ]; then
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
MODEL_PATH="./offline_assets/models/Qwen--Qwen2.5-7B-Instruct"
DATASET_PATH="./offline_assets/datasets/openbmb--UltraFeedback"
OUTPUT_DIR="./output_offline"

echo ""
echo "训练配置:"
echo "  模型: $MODEL_PATH"
echo "  数据集: $DATASET_PATH"
echo "  输出: $OUTPUT_DIR"
echo ""

# 询问是否继续
read -p "是否开始训练? (y/n): " CONTINUE
if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="

python train_rlhf.py \
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
