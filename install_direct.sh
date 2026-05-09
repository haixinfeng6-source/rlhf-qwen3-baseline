#!/bin/bash

# 直接安装脚本 - 不使用虚拟环境
# 适用于磁盘空间受限的情况

set -e

echo "=================================================="
echo "RLHF 环境直接安装（无虚拟环境）"
echo "=================================================="

# 检查当前Python
echo -e "\n当前Python环境:"
which python
python --version

# 检查磁盘空间
echo -e "\n磁盘空间检查:"
df -h .

# 清理缓存以释放空间
echo -e "\n清理缓存..."
pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true
rm -rf ~/.cache/huggingface 2>/dev/null || true

# 设置临时目录（如果有其他磁盘）
# 如果 /data 或 /scratch 有空间，可以设置:
# export TMPDIR=/data/$USER/tmp
# mkdir -p $TMPDIR

# ============================================
# 步骤1: 检查已安装的包
# ============================================

echo -e "\n步骤1: 检查已安装的包..."

# 检查PyTorch
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    echo "  PyTorch: $TORCH_VERSION"
    echo "  CUDA可用: $CUDA_AVAILABLE"
else
    echo "  PyTorch: 未安装"
fi

# 检查其他包
for pkg in transformers trl datasets accelerate peft; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)")
        echo "  $pkg: $VERSION"
    else
        echo "  $pkg: 未安装"
    fi
done

# ============================================
# 步骤2: 安装PyTorch
# ============================================

echo -e "\n步骤2: 安装PyTorch..."

read -p "选择CUDA版本: 1) 12.8  2) 12.1  3) 11.8  4) CPU only [默认: 1]: " CUDA_CHOICE

case $CUDA_CHOICE in
    2)
        CUDA_VERSION="12.1"
        TORCH_VERSION="2.5.1"
        TORCH_URL="https://download.pytorch.org/whl/cu121"
        ;;
    3)
        CUDA_VERSION="11.8"
        TORCH_VERSION="2.1.0"
        TORCH_URL="https://download.pytorch.org/whl/cu118"
        ;;
    4)
        CUDA_VERSION="cpu"
        TORCH_VERSION="2.7.0"
        TORCH_URL="https://download.pytorch.org/whl/cpu"
        ;;
    1 | *)
        CUDA_VERSION="12.8"
        TORCH_VERSION="2.7.0"
        TORCH_URL="https://download.pytorch.org/whl/cu128"
        ;;
esac

echo "安装PyTorch $TORCH_VERSION (CUDA $CUDA_VERSION)..."

if [[ "$CUDA_VERSION" != "cpu" ]]; then
    pip install --no-cache-dir torch==${TORCH_VERSION}+cu${CUDA_VERSION//./} \
        --index-url "$TORCH_URL" \
        --user
else
    pip install --no-cache-dir torch==${TORCH_VERSION} \
        --index-url "$TORCH_URL" \
        --user
fi

echo "✓ PyTorch安装完成"

# ============================================
# 步骤3: 安装核心依赖
# ============================================

echo -e "\n步骤3: 安装核心依赖..."

# 使用--no-cache-dir节省空间
# 使用--user安装到用户目录

pip install --no-cache-dir --user \
    transformers==4.46.0 \
    datasets==3.1.0 \
    accelerate==1.0.0 \
    trl==0.11.4 \
    peft==0.13.2

echo "✓ 核心依赖安装完成"

# ============================================
# 步骤4: 安装辅助依赖
# ============================================

echo -e "\n步骤4: 安装辅助依赖..."

pip install --no-cache-dir --user \
    numpy \
    pandas \
    tqdm \
    PyYAML \
    requests

echo "✓ 辅助依赖安装完成"

# ============================================
# 步骤5: 验证安装
# ============================================

echo -e "\n步骤5: 验证安装..."

# 检查PyTorch
python << 'EOF'
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
EOF

# 检查TRL
python << 'EOF'
try:
    from trl import PPOConfig, PPOTrainer
    from trl.models import AutoModelForCausalLMWithValueHead
    print("✓ TRL导入成功")
except ImportError as e:
    print(f"✗ TRL导入失败: {e}")
    exit(1)
EOF

# 检查transformers
python << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
print(f"✓ transformers导入成功 (版本: {transformers.__version__})")
EOF

# 检查其他包
python << 'EOF'
import datasets
import peft
import accelerate

print(f"✓ datasets: {datasets.__version__}")
print(f"✓ peft: {peft.__version__}")
print(f"✓ accelerate: {accelerate.__version__}")
EOF

# ============================================
# 步骤6: 保存安装信息
# ============================================

echo -e "\n步骤6: 保存安装信息..."

pip list > installed_packages.txt
echo "✓ 已安装包列表: installed_packages.txt"

python << EOF > environment_info.json
import sys
import torch
import json
import platform

info = {
    'python_version': sys.version,
    'python_path': sys.executable,
    'platform': platform.platform(),
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
}

print(json.dumps(info, indent=2))
EOF

echo "✓ 环境信息: environment_info.json"

# ============================================
# 完成
# ============================================

echo -e "\n=================================================="
echo "安装完成！"
echo "=================================================="

echo -e "\n已安装包:"
cat installed_packages.txt | grep -E "(torch|transformers|trl|datasets|peft|accelerate)"

echo -e "\n下一步:"
echo "  1. 下载离线资源: python download_assets.py"
echo "  2. 运行训练: python train_rlhf_simple.py"
echo "  3. 提交作业: sbatch submit_train.slurm"

echo -e "\n注意事项:"
echo "  - 使用--user安装，包在 ~/.local/"
echo "  - 不需要激活虚拟环境"
echo "  - 直接运行python命令即可"

echo -e "\n配置成功！"