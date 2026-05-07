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
#SBATCH --partition=gpu  # 根据集群配置调整
#SBATCH --qos=normal     # 根据集群配置调整

echo "=========================================="
echo "RLHF 训练作业开始"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_JOB_NODELIST"
echo "开始时间: $(date)"
echo "=========================================="

# 加载必要的模块
module purge
module load cuda/11.8
module load python/3.10
module load gcc/9.3.0

echo "模块加载完成"

# 激活Python环境
source ~/rlhf-cluster/bin/activate
echo "Python环境已激活"

# 设置工作目录
cd /path/to/rlhf-qwen3-baseline
echo "工作目录: $(pwd)"

# 检查离线资源
echo "检查离线资源..."
if [ ! -d "offline_assets/models/Qwen--Qwen3-8B" ]; then
    echo "[错误] 模型文件不存在"
    exit 1
fi

if [ ! -d "offline_assets/datasets/openbmb--UltraFeedback" ]; then
    echo "[错误] 数据集文件不存在"
    exit 1
fi
echo "✓ 离线资源检查通过"

# 设置环境变量
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

echo "环境变量设置完成"

# 显示GPU信息
echo "GPU信息:"
nvidia-smi
echo ""

# 训练参数
MODEL_PATH="./offline_assets/models/Qwen--Qwen3-8B"
DATASET_PATH="./offline_assets/datasets/openbmb--UltraFeedback"
OUTPUT_DIR="./output_rlhf_${SLURM_JOB_ID}"
NUM_GPUS=4

echo "训练配置:"
echo "  模型: $MODEL_PATH"
echo "  数据集: $DATASET_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  GPU数量: $NUM_GPUS"
echo ""

# 开始训练
echo "=========================================="
echo "开始训练..."
echo "=========================================="

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_rlhf_cluster_fixed.py \
    --model_name="$MODEL_PATH" \
    --dataset_name="$DATASET_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --use_lora \
    --lora_r=16 \
    --lora_alpha=32 \
    --gradient_checkpointing \
    --logging_steps=10 \
    --save_steps=500 \
    --max_length=512 \
    --max_samples=100  # 测试用，成功后可以移除或增加

TRAINING_EXIT_CODE=$?

echo "=========================================="
echo "训练结束"
echo "结束时间: $(date)"
echo "退出代码: $TRAINING_EXIT_CODE"
echo "=========================================="

# 检查输出
if [ $TRAINING_EXIT_CODE -eq 0 ] && [ -d "$OUTPUT_DIR" ]; then
    echo "训练成功完成!"
    echo "输出目录内容:"
    ls -la "$OUTPUT_DIR"
    
    # 检查模型文件
    if [ -d "$OUTPUT_DIR/final_model" ]; then
        echo "模型文件已保存: $OUTPUT_DIR/final_model"
        du -sh "$OUTPUT_DIR/final_model"
    fi
else
    echo "训练失败或未完成"
    echo "请检查错误日志: rlhf-${SLURM_JOB_ID}.err"
fi

echo "作业完成"