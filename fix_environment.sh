#!/bin/bash

# 修复环境脚本
# 降级到兼容的版本组合

echo "=========================================="
echo "修复 RLHF 训练环境"
echo "=========================================="

echo "当前环境:"
python --version
pip --version

echo ""
echo "1. 卸载当前版本..."
pip uninstall -y trl peft transformers torch

echo ""
echo "2. 安装兼容版本..."
echo "注意: 由于 CUDA 驱动太旧，使用 CPU 版本"

# 安装 CPU 版本的 PyTorch（因为 CUDA 驱动太旧）
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 安装其他兼容版本
pip install transformers==4.36.0
pip install trl==0.11.0
pip install peft==0.7.0
pip install datasets==2.14.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.0
pip install rich==13.7.0
pip install tensorboard==2.14.0

echo ""
echo "3. 验证安装..."
pip list | grep -E "(torch|transformers|trl|peft|datasets)"

echo ""
echo "4. 测试导入..."
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')

# 测试 TRL 导入
try:
    from trl.models import AutoModelForCausalLMWithValueHead
    print('✓ TRL 0.11.0 导入成功')
except ImportError:
    try:
        from trl import AutoModelForCausalLMWithValueHead
        print('✓ TRL 0.10.0 导入成功')
    except ImportError as e:
        print(f'✗ TRL 导入失败: {e}')

# 测试 PEFT 导入
try:
    from peft import LoraConfig
    print('✓ PEFT 导入成功')
except ImportError as e:
    print(f'✗ PEFT 导入失败: {e}')

# 测试 TRL 训练器
try:
    from trl import PPOConfig, PPOTrainer
    print('✓ PPOConfig, PPOTrainer 导入成功')
except ImportError as e:
    print(f'✗ PPOConfig, PPOTrainer 导入失败: {e}')
"

echo ""
echo "=========================================="
echo "修复完成!"
echo "=========================================="
echo "下一步:"
echo "1. 运行测试: python direct_import_test.py"
echo "2. 如果测试通过: python run_single_process.py"
echo "3. 注意: 由于 CUDA 不可用，训练将在 CPU 上进行"
echo "=========================================="