#!/bin/bash

# bitsandbytes错误快速修复脚本

echo "=========================================="
echo "bitsandbytes/triton错误快速修复"
echo "=========================================="

echo "当前错误: ModuleNotFoundError: No module named 'triton.ops'"

# 选项1: 禁用bitsandbytes（最简单）
echo -e "\n选项1: 禁用bitsandbytes（推荐）"
echo "------------------------------------------"
echo "运行无需bitsandbytes的版本:"
echo ""
echo "python train_rlhf_no_bitsandbytes.py \\"
echo "  --model_name='./offline_assets/models/Qwen--Qwen3-8B' \\"
echo "  --dataset_name='./offline_assets/datasets/openbmb--UltraFeedback' \\"
echo "  --output_dir='./test_no_bnb' \\"
echo "  --max_samples=5"
echo ""
read -p "是否运行此选项? (y/n): " RUN_OPTION1
if [ "$RUN_OPTION1" = "y" ] || [ "$RUN_OPTION1" = "Y" ]; then
    python train_rlhf_no_bitsandbytes.py \
      --model_name='./offline_assets/models/Qwen--Qwen3-8B' \
      --dataset_name='./offline_assets/datasets/openbmb--UltraFeedback' \
      --output_dir='./test_no_bnb' \
      --max_samples=5
    exit 0
fi

# 选项2: 修复安装
echo -e "\n选项2: 修复bitsandbytes安装"
echo "------------------------------------------"
echo "步骤:"
echo "1. 检查CUDA是否可用"
echo "2. 重新安装bitsandbytes"
echo "3. 安装triton"
echo ""
read -p "是否继续? (y/n): " RUN_OPTION2
if [ "$RUN_OPTION2" = "y" ] || [ "$RUN_OPTION2" = "Y" ]; then
    # 检查CUDA
    echo -e "\n检查CUDA..."
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
    
    # 重新安装bitsandbytes
    echo -e "\n重新安装bitsandbytes..."
    pip uninstall bitsandbytes -y
    
    # 根据CUDA情况安装
    python -c "
import torch
if torch.cuda.is_available():
    print('CUDA可用，安装GPU版本')
else:
    print('CUDA不可用，安装CPU版本')
" > /tmp/cuda_check.txt
    
    if grep -q "CUDA可用" /tmp/cuda_check.txt; then
        echo "安装bitsandbytes GPU版本..."
        pip install bitsandbytes==0.41.0
    else
        echo "安装bitsandbytes CPU版本..."
        pip install bitsandbytes-cpu
    fi
    
    # 安装triton
    echo -e "\n安装triton..."
    pip install triton==2.0.0
    
    # 测试
    echo -e "\n测试修复..."
    python -c "
try:
    import triton.ops
    print('✅ triton.ops 导入成功')
except ImportError as e:
    print(f'❌ triton.ops 导入失败: {e}')

try:
    import bitsandbytes
    print(f'✅ bitsandbytes 导入成功 (版本: {bitsandbytes.__version__})')
except ImportError as e:
    print(f'❌ bitsandbytes 导入失败: {e}')
"
    
    echo -e "\n修复完成！"
fi

# 选项3: 使用详细诊断
echo -e "\n选项3: 运行详细诊断"
echo "------------------------------------------"
read -p "是否运行详细诊断? (y/n): " RUN_OPTION3
if [ "$RUN_OPTION3" = "y" ] || [ "$RUN_OPTION3" = "Y" ]; then
    python fix_triton_error.py
fi

# 总结
echo -e "\n=========================================="
echo "总结"
echo "=========================================="
echo "问题原因:"
echo "  bitsandbytes需要triton.ops模块，但可能:"
echo "  1. triton未安装"
echo "  2. triton版本不兼容"
echo "  3. bitsandbytes安装损坏"
echo ""
echo "推荐解决方案:"
echo "  1. 禁用bitsandbytes（最简单）"
echo "  2. 重新安装bitsandbytes和triton"
echo "  3. 创建干净环境"
echo ""
echo "立即测试命令:"
echo "  python train_rlhf_no_bitsandbytes.py --max_samples=2"
echo ""
echo "如果仍然有问题，查看详细文档:"
echo "  python fix_triton_error.py"