#!/bin/bash

# RLHF 集群环境修复脚本
# 解决以下问题:
# 1. CUDA驱动太旧 (12080)
# 2. TRL和Transformers版本不兼容
# 3. Python 3.13兼容性问题

echo "=========================================="
echo "RLHF 集群环境修复"
echo "=========================================="

# 检查当前环境
echo "1. 检查当前环境..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查NVIDIA驱动
echo -e "\n2. 检查NVIDIA驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
    echo "⚠️ 如果驱动版本低于12090，建议联系管理员更新"
else
    echo "nvidia-smi 未找到"
fi

# 检查包版本
echo -e "\n3. 检查包版本..."
python -c "
import importlib.metadata
pkgs = ['torch', 'transformers', 'trl', 'datasets', 'accelerate', 'peft']
for pkg in pkgs:
    try:
        v = importlib.metadata.version(pkg)
        print(f'{pkg:20} {v}')
    except:
        print(f'{pkg:20} 未安装')
"

# 方案1: 创建干净环境（推荐）
echo -e "\n4. 推荐方案: 创建干净环境"
echo "------------------------------------------"
echo "步骤1: 创建新环境"
echo "  python3.10 -m venv ~/rlhf-clean"
echo "  source ~/rlhf-clean/bin/activate"
echo ""
echo "步骤2: 安装兼容版本"
echo "  pip install torch==2.1.0 transformers==4.36.0 trl==0.11.0"
echo "  pip install datasets==2.14.0 accelerate==0.25.0 peft==0.7.0"
echo "  pip install rich tensorboard"
echo ""
echo "步骤3: 测试导入"
echo "  python -c \"from trl import PPOConfig, PPOTrainer; print('导入成功')\""

# 方案2: 修复当前环境
echo -e "\n5. 备用方案: 修复当前环境"
echo "------------------------------------------"
read -p "是否尝试修复当前环境? (y/n): " FIX_CURRENT

if [ "$FIX_CURRENT" = "y" ] || [ "$FIX_CURRENT" = "Y" ]; then
    echo "开始修复..."
    
    # 备份当前requirements
    echo "备份当前包版本..."
    pip freeze > requirements_backup.txt
    
    # 尝试降级关键包
    echo "降级关键包..."
    pip install --force-reinstall torch==2.1.0 transformers==4.36.0 trl==0.11.0
    
    # 测试修复
    echo "测试修复..."
    python fix_import_issue.py
    
    if [ $? -eq 0 ]; then
        echo "✅ 修复成功！"
    else
        echo "❌ 修复失败，建议使用干净环境方案"
    fi
fi

# 方案3: 使用CPU模式
echo -e "\n6. 紧急方案: 使用CPU模式测试"
echo "------------------------------------------"
echo "如果GPU不可用，可以使用CPU模式测试流程:"
echo ""
echo "步骤1: 运行兼容性测试"
echo "  python train_rlhf_compatible.py \\"
echo "    --model_name='./offline_assets/models/Qwen--Qwen3-8B' \\"
echo "    --dataset_name='./offline_assets/datasets/openbmb--UltraFeedback' \\"
echo "    --output_dir='./test_cpu' \\"
echo "    --max_samples=5"
echo ""
echo "步骤2: 验证环境"
echo "  如果测试通过，说明代码逻辑正确"
echo "  然后在GPU集群上运行完整训练"

# 集群作业建议
echo -e "\n7. 集群作业建议"
echo "------------------------------------------"
echo "在SLURM作业脚本中添加以下内容:"
echo ""
echo "# 设置环境变量"
echo "export HF_HUB_OFFLINE=1"
echo "export TRANSFORMERS_OFFLINE=1"
echo "export HF_DATASETS_OFFLINE=1"
echo ""
echo "# 使用CPU模式回退"
echo "export CUDA_VISIBLE_DEVICES=''  # 如果需要强制CPU模式"
echo ""
echo "# 运行兼容性脚本"
echo "python train_rlhf_compatible.py \\"
echo "  --model_name='\${PWD}/offline_assets/models/Qwen--Qwen3-8B' \\"
echo "  --dataset_name='\${PWD}/offline_assets/datasets/openbmb--UltraFeedback' \\"
echo "  --output_dir='./output_\${SLURM_JOB_ID}' \\"
echo "  --max_samples=100  # 测试用"

# 总结
echo -e "\n=========================================="
echo "总结"
echo "=========================================="
echo "主要问题:"
echo "  1. CUDA驱动版本12080太旧"
echo "  2. TRL 0.11.4与Transformers 4.46.3不兼容"
echo "  3. 需要特定版本组合"
echo ""
echo "推荐解决方案:"
echo "  1. 创建干净Python 3.10环境"
echo "  2. 安装固定版本包 (见requirements_cluster.txt)"
echo "  3. 在GPU集群上测试"
echo ""
echo "紧急测试:"
echo "  使用 train_rlhf_compatible.py 进行CPU模式测试"
echo ""
echo "文件说明:"
echo "  - fix_import_issue.py: 导入修复工具"
echo "  - train_rlhf_compatible.py: 兼容性训练脚本"
echo "  - requirements_cluster.txt: 推荐版本"
echo "  - run_cluster_offline.sh: 集群启动脚本"

echo -e "\n修复完成！"