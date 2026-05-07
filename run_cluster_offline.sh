#!/bin/bash

# RLHF 集群离线训练启动脚本
# 专为无网络GPU集群设计

echo "=========================================="
echo "RLHF 集群离线训练"
echo "=========================================="

# 检查离线资源
echo "检查离线资源..."
if [ ! -d "offline_assets/models/Qwen--Qwen3-8B" ]; then
    echo "[错误] 模型文件不存在: offline_assets/models/Qwen--Qwen3-8B"
    echo "请确保离线资源已正确传输到集群"
    exit 1
fi

if [ ! -d "offline_assets/datasets/openbmb--UltraFeedback" ]; then
    echo "[错误] 数据集文件不存在: offline_assets/datasets/openbmb--UltraFeedback"
    echo "请确保离线资源已正确传输到集群"
    exit 1
fi

echo "✓ 离线资源检查通过"

# 设置环境变量 - 强制离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

# 设置NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据集群网络接口调整
export NCCL_IB_DISABLE=0

# 设置PyTorch分布式环境
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

# 训练配置
MODEL_PATH="./offline_assets/models/Qwen--Qwen3-8B"
DATASET_PATH="./offline_assets/datasets/openbmb--UltraFeedback"
OUTPUT_DIR="./output_cluster_$(date +%Y%m%d_%H%M%S)"

# GPU数量 - 根据集群配置调整
NUM_GPUS=4

echo ""
echo "训练配置:"
echo "  模型: $MODEL_PATH"
echo "  数据集: $DATASET_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  GPU数量: $NUM_GPUS"
echo "  离线模式: 已启用"
echo ""

# 环境检查
echo "环境检查..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 检查TRL版本
echo "检查TRL版本..."
python -c "
try:
    import importlib.metadata
    trl_version = importlib.metadata.version('trl')
    print(f'TRL版本: {trl_version}')
except:
    print('无法获取TRL版本')
"

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

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_rlhf_cluster_fixed.py \
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
    --gradient_checkpointing \
    --logging_steps=5 \
    --save_steps=100 \
    --max_length=512 \
    --max_samples=100  # 测试用，减少样本数

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: $OUTPUT_DIR"
echo "日志文件: $OUTPUT_DIR/logs"
echo ""

# 检查输出
if [ -d "$OUTPUT_DIR" ]; then
    echo "输出目录内容:"
    ls -la "$OUTPUT_DIR"
fi