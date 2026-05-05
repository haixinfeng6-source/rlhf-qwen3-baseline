#!/bin/bash

# 快速开始脚本 - 一键安装和启动训练

echo "=========================================="
echo "RLHF Training Quick Start"
echo "=========================================="

# 1. 检查 Python 环境
echo "Step 1: Checking Python environment..."
python --version
if [ $? -ne 0 ]; then
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi

# 2. 安装依赖
echo ""
echo "Step 2: Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# 3. 检查 GPU
echo ""
echo "Step 3: Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "Warning: GPU check failed. Training may not work properly."
fi

# 4. 创建输出目录
echo ""
echo "Step 4: Creating output directory..."
mkdir -p outputs
mkdir -p logs

# 5. 选择训练方法
echo ""
echo "Step 5: Select training method:"
echo "1) PPO (Proximal Policy Optimization)"
echo "2) GRPO (Group Relative Policy Optimization) - Faster, less memory"
read -p "Enter choice [1-2]: " method_choice

if [ "$method_choice" = "2" ]; then
    METHOD="grpo"
else
    METHOD="ppo"
fi

# 6. 选择 GPU 数量
echo ""
read -p "Enter number of GPUs [4 or 8]: " num_gpus

if [ "$num_gpus" != "4" ] && [ "$num_gpus" != "8" ]; then
    echo "Invalid GPU count. Using 4 GPUs."
    num_gpus=4
fi

# 7. 设置环境变量
if [ "$num_gpus" = "8" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# 8. 开始训练
echo ""
echo "=========================================="
echo "Starting training with:"
echo "  Method: ${METHOD}"
echo "  GPUs: ${num_gpus}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="
echo ""

# 修改启动脚本中的参数
sed -i "s/METHOD=\".*\"/METHOD=\"${METHOD}\"/" run_multi_gpu.sh
sed -i "s/NUM_GPUS=.*/NUM_GPUS=${num_gpus}/" run_multi_gpu.sh

# 启动训练
bash run_multi_gpu.sh

echo ""
echo "=========================================="
echo "Training script completed!"
echo "Check outputs/ directory for results"
echo "=========================================="
