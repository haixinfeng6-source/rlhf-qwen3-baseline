#!/bin/bash

# RLHF 训练最终解决方案
# 解决所有环境问题

echo "=========================================="
echo "RLHF 训练 - 最终解决方案"
echo "=========================================="

echo "检测到的问题:"
echo "1. ✅ PyTorch 已安装 (2.11.0)"
echo "2. ❌ CUDA 驱动太旧 (12080)"
echo "3. ❌ triton.ops 不存在"
echo "4. ❌ 包版本冲突 (xtuner, lmdeploy)"
echo "5. ❌ bitsandbytes GPU 支持缺失"
echo ""

echo "解决方案选择:"
echo ""
echo "方案 A: 快速修复 (立即尝试)"
echo "----------------------------------------"
echo "1. 安装 triton:"
echo "   pip install triton"
echo ""
echo "2. 设置环境变量:"
echo "   export BITSANDBYTES_NOWELCOME=1"
echo "   export CUDA_VISIBLE_DEVICES=''"
echo ""
echo "3. 运行修复脚本:"
echo "   python fix_triton.py"
echo ""
echo "4. 测试:"
echo "   python train_simple.py"
echo ""

echo "方案 B: 独立环境 (推荐)"
echo "----------------------------------------"
echo "1. 创建 conda 环境:"
echo "   conda create -n rlhf-final python=3.10 -y"
echo "   conda activate rlhf-final"
echo ""
echo "2. 安装最小依赖:"
cat > requirements_final.txt << 'EOF'
# RLHF 最小依赖
torch==2.1.0
transformers==4.36.0
trl==0.11.0
datasets==2.14.0
accelerate==0.25.0
peft==0.7.0
# CPU 版本，避免 GPU 问题
EOF
echo "   已创建: requirements_final.txt"
echo ""
echo "3. 安装:"
echo "   pip install -r requirements_final.txt"
echo "   pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu"
echo ""

echo "方案 C: Docker (最干净)"
echo "----------------------------------------"
echo "1. 安装 Docker"
echo "2. 使用预构建镜像:"
echo "   docker pull huggingface/transformers-pytorch-gpu:latest"
echo "3. 运行:"
echo "   docker run --gpus all -it huggingface/transformers-pytorch-gpu"
echo ""

echo "方案 D: 云服务 (无需本地配置)"
echo "----------------------------------------"
echo "1. Google Colab:"
echo "   https://colab.research.google.com/"
echo "2. 上传代码，使用免费 GPU"
echo "3. 无需解决环境问题"
echo ""

echo "=========================================="
echo "立即执行方案 A"
echo "=========================================="

# 执行方案 A
echo "执行快速修复..."
echo ""

echo "1. 安装 triton..."
pip install triton --quiet 2>/dev/null || echo "安装失败，尝试继续"

echo ""
echo "2. 设置环境变量..."
export BITSANDBYTES_NOWELCOME=1
export CUDA_VISIBLE_DEVICES=''

echo ""
echo "3. 创建修复文件..."
cat > quick_fix.py << 'EOF'
import os
import sys

print("快速修复...")

# 设置环境
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 尝试导入
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    
    # 检查 triton
    try:
        import triton
        print(f"✓ triton: {triton.__version__}")
    except ImportError:
        print("⚠️  triton 未安装，但不一定需要")
    
    print("\n✓ 环境准备完成")
    print("可以尝试运行训练")
    
except Exception as e:
    print(f"✗ 错误: {e}")

print("\n如果仍有问题，建议使用方案 B (独立环境)")
EOF

echo "已创建: quick_fix.py"
echo ""
echo "4. 运行测试..."
python quick_fix.py

echo ""
echo "=========================================="
echo "创建一键修复脚本"
echo "=========================================="

cat > one_click_fix.sh << 'EOF'
#!/bin/bash
# RLHF 一键修复脚本

echo "RLHF 环境一键修复..."
echo ""

# 创建独立目录
mkdir -p rlhf_workspace
cd rlhf_workspace

# 创建虚拟环境
python3.10 -m venv venv_rlhf
source venv_rlhf/bin/activate

# 安装依赖
pip install --upgrade pip
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.36.0
pip install trl==0.11.0
pip install datasets==2.14.0
pip install accelerate==0.25.0

echo ""
echo "✓ 环境创建完成!"
echo ""
echo "使用方法:"
echo "1. cd rlhf_workspace"
echo "2. source venv_rlhf/bin/activate"
echo "3. 复制训练代码到此目录"
echo "4. 运行训练"
echo ""
echo "注意: 使用 CPU 训练，速度较慢"
EOF

chmod +x one_click_fix.sh
echo "已创建: one_click_fix.sh"

echo ""
echo "=========================================="
echo "总结"
echo "=========================================="
echo "由于环境问题复杂，建议:"
echo ""
echo "1. 立即方案: 运行 one_click_fix.sh"
echo "   创建完全独立的工作空间"
echo ""
echo "2. 或者使用云服务: Google Colab"
echo "   无需解决本地环境问题"
echo ""
echo "3. 长期方案: 更新 NVIDIA 驱动"
echo "   然后使用独立环境"
echo ""
echo "=========================================="
echo "命令汇总"
echo "=========================================="
echo "快速测试: python train_simple.py"
echo "一键修复: ./one_click_fix.sh"
echo "独立环境: conda create -n rlhf python=3.10"
echo "云服务: https://colab.research.google.com/"
echo "驱动更新: http://www.nvidia.com/Download/"
echo "=========================================="