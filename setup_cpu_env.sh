#!/bin/bash

# CPU节点环境配置脚本
# 在CPU节点上运行，配置完整的训练环境

set -e  # 遇到错误立即退出

echo "=================================================="
echo "RLHF 项目 - CPU节点环境配置"
echo "=================================================="

# ============================================
# 步骤1: 检查系统环境
# ============================================

echo -e "\n步骤1: 检查系统环境..."
echo "Python版本: $(python --version)"
echo "当前目录: $(pwd)"

# 检查Python版本
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python版本号: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" ]] && [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "⚠️  警告: 推荐使用Python 3.10或3.11"
    read -p "是否继续? (y/n): " CONTINUE
    if [[ "$CONTINUE" != "y" ]]; then
        exit 1
    fi
fi

# ============================================
# 步骤2: 创建虚拟环境
# ============================================

echo -e "\n步骤2: 创建虚拟环境..."

VENV_DIR="rlhf-env"

if [ -d "$VENV_DIR" ]; then
    echo "虚拟环境已存在: $VENV_DIR"
    read -p "是否删除并重新创建? (y/n): " RECREATE
    if [[ "$RECREATE" == "y" ]]; then
        echo "删除旧环境..."
        rm -rf "$VENV_DIR"
    else
        echo "使用现有环境"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "创建虚拟环境: $VENV_DIR"
    python -m venv "$VENV_DIR"
    echo "✓ 虚拟环境创建成功"
fi

# 激活环境
echo "激活虚拟环境..."
source "$VENV_DIR/bin/activate"
echo "✓ 环境已激活"

# ============================================
# 步骤3: 安装PyTorch
# ============================================

echo -e "\n步骤3: 安装PyTorch..."

read -p "选择CUDA版本: 1) 11.8  2) 12.1  3) 12.8  4) CPU only [默认: 3]: " CUDA_CHOICE

case $CUDA_CHOICE in
    1)
        CUDA_VERSION="11.8"
        TORCH_VERSION="2.1.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        ;;
    2)
        CUDA_VERSION="12.1"
        TORCH_VERSION="2.5.1"
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        ;;
    4)
        CUDA_VERSION="cpu"
        TORCH_VERSION="2.7.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        ;;
    3 | *)
        CUDA_VERSION="12.8"
        TORCH_VERSION="2.7.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        ;;
esac

echo "安装PyTorch $TORCH_VERSION (CUDA $CUDA_VERSION)..."
if [[ "$CUDA_VERSION" != "cpu" ]]; then
    pip install torch==${TORCH_VERSION}+cu${CUDA_VERSION//./} --index-url "$TORCH_INDEX"
else
    pip install torch==${TORCH_VERSION}+cpu --index-url "$TORCH_INDEX"
fi

echo "✓ PyTorch安装成功"

# ============================================
# 步骤4: 安装核心依赖
# ============================================

echo -e "\n步骤4: 安装核心依赖..."

echo "安装transformers..."
pip install transformers==4.46.0

echo "安装TRL..."
pip install trl==0.11.4

echo "安装datasets..."
pip install datasets==3.1.0

echo "安装accelerate..."
pip install accelerate==1.0.0

echo "安装PEFT..."
pip install peft==0.13.2

echo "安装其他依赖..."
pip install numpy pandas tqdm PyYAML requests

echo "✓ 核心依赖安装完成"

# ============================================
# 步骤5: 验证安装
# ============================================

echo -e "\n步骤5: 验证安装..."

echo "检查PyTorch..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"

echo "检查TRL..."
python -c "
try:
    from trl import PPOConfig, PPOTrainer
    from trl.models import AutoModelForCausalLMWithValueHead
    print('✓ TRL导入成功')
except ImportError as e:
    print(f'✗ TRL导入失败: {e}')
    exit(1)
"

echo "检查transformers..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('✓ transformers导入成功')
"

echo "检查其他包..."
python -c "
import datasets
import peft
import accelerate
print('✓ 其他包导入成功')
"

# ============================================
# 步骤6: 下载离线资源
# ============================================

echo -e "\n步骤6: 下载离线资源..."

read -p "是否下载模型和数据集? (y/n): " DOWNLOAD_ASSETS

if [[ "$DOWNLOAD_ASSETS" == "y" ]]; then
    if [ -f "download_assets.py" ]; then
        echo "运行下载脚本..."
        python download_assets.py
        echo "✓ 离线资源下载完成"
    else
        echo "⚠️  未找到download_assets.py"
        echo "请手动下载模型和数据集"
    fi
else
    echo "跳过下载"
fi

# ============================================
# 步骤7: 保存环境信息
# ============================================

echo -e "\n步骤7: 保存环境信息..."

pip freeze > requirements_installed.txt
echo "✓ 已保存已安装包列表: requirements_installed.txt"

python -c "
import sys
import torch
import platform

info = {
    'python_version': sys.version,
    'platform': platform.platform(),
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
}

import json
with open('environment_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print('✓ 已保存环境信息: environment_info.json')
"

# ============================================
# 步骤8: 创建启动脚本
# ============================================

echo -e "\n步骤8: 创建启动脚本..."

cat > run_train.sh << 'EOF'
#!/bin/bash
# RLHF训练启动脚本

# 激活环境
source rlhf-env/bin/activate

# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 运行训练
python train_rlhf_simple.py --config config_cluster.yaml "$@"
EOF

chmod +x run_train.sh
echo "✓ 已创建启动脚本: run_train.sh"

# ============================================
# 完成
# ============================================

echo -e "\n=================================================="
echo "环境配置完成！"
echo "=================================================="

echo -e "\n环境信息:"
echo "  虚拟环境: $(pwd)/$VENV_DIR"
echo "  Python: $(which python)"
echo "  已安装包: requirements_installed.txt"
echo "  环境信息: environment_info.json"

echo -e "\n下一步:"
echo "  1. 测试环境: python train_rlhf_simple.py"
echo "  2. 运行训练: bash run_train.sh"
echo "  3. 打包环境: tar -czf rlhf-env.tar.gz rlhf-env/"

echo -e "\n如需打包环境传输到集群:"
echo "  tar -czf rlhf-project.tar.gz rlhf-qwen3-baseline/"

echo -e "\n配置成功！"