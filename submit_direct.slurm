#!/bin/bash
#SBATCH --job-name=rlhf-direct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=rlhf-direct-%j.out
#SBATCH --error=rlhf-direct-%j.err

# ============================================
# RLHF训练作业 - 直接安装版本（无虚拟环境）
# ============================================

echo "=================================================="
echo "RLHF 训练作业"
echo "=================================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_JOB_NODELIST"
echo "开始时间: $(date)"
echo "=================================================="

# 加载模块（根据集群配置）
module load cuda/12.8
module load python/3.10

# 注意: 不需要激活虚拟环境
# 使用直接安装的包（--user安装）

# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 设置环境变量
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

# 进入工作目录
cd ~/rlhf-qwen3-baseline

# 显示环境信息
echo ""
echo "环境信息:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
nvidia-smi

# 运行训练
echo ""
echo "=================================================="
echo "开始训练..."
echo "=================================================="

python train_rlhf_simple.py \
    --config config_cluster.yaml \
    --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
    --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
    --output_dir="./output_${SLURM_JOB_ID}"

TRAIN_EXIT_CODE=$?

# 训练结束
echo ""
echo "=================================================="
echo "训练结束"
echo "结束时间: $(date)"
echo "退出代码: $TRAIN_EXIT_CODE"
echo "=================================================="

# 检查输出
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成！"
    
    if [ -d "./output_${SLURM_JOB_ID}" ]; then
        echo ""
        echo "输出目录内容:"
        ls -lh "./output_${SLURM_JOB_ID}"
    fi
else
    echo "❌ 训练失败"
    echo "查看错误日志: rlhf-direct-${SLURM_JOB_ID}.err"
fi

echo ""
echo "作业完成"