#!/bin/bash

# 离线安装脚本
# 在开发机上运行此脚本安装依赖

echo "=========================================="
echo "离线安装 Python 依赖"
echo "=========================================="

# 检查 wheels 目录
if [ ! -d "offline_assets/wheels" ]; then
    echo "✗ 未找到 offline_assets/wheels 目录"
    echo "请确保已将 offline_assets 目录上传到开发机"
    exit 1
fi

# 检查 requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "✗ 未找到 requirements.txt 文件"
    exit 1
fi

echo "从 offline_assets/wheels 目录安装依赖..."
pip install --no-index --find-links=offline_assets/wheels -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ 依赖安装完成"
else
    echo "✗ 依赖安装失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="

# 验证 PyTorch
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo "✗ PyTorch 未安装"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null

# 验证 Transformers
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" 2>/dev/null || echo "✗ Transformers 未安装"

# 验证 TRL
python -c "import trl; print(f'✓ TRL: {trl.__version__}')" 2>/dev/null || echo "✗ TRL 未安装"

# 验证 PEFT
python -c "import peft; print(f'✓ PEFT: {peft.__version__}')" 2>/dev/null || echo "✗ PEFT 未安装"

# 验证 Datasets
python -c "import datasets; print(f'✓ Datasets: {datasets.__version__}')" 2>/dev/null || echo "✗ Datasets 未安装"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 测试离线资源:"
echo "   python test_offline_assets.py"
echo ""
echo "2. 启动训练:"
echo "   python train_rlhf.py \\"
echo "     --model_name=./offline_assets/models/Qwen--Qwen2.5-7B-Instruct \\"
echo "     --dataset_name=./offline_assets/datasets/openbmb--UltraFeedback \\"
echo "     --output_dir=./output_offline"
echo ""
