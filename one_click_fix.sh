#!/bin/bash

# 一键修复所有RLHF环境问题

echo "=================================================="
echo "RLHF 环境一键修复"
echo "=================================================="

echo "正在修复的问题:"
echo "1. bitsandbytes.__spec__ is None"
echo "2. is_torch_mlu_available 导入错误"
echo "3. TRL导入问题"
echo "4. 包版本冲突"
echo ""

# 步骤1: 备份当前环境
echo "步骤1: 备份当前环境..."
pip freeze > requirements_backup_$(date +%Y%m%d_%H%M%S).txt
echo "✅ 环境已备份"

# 步骤2: 运行终极修复
echo -e "\n步骤2: 运行终极修复诊断..."
python ultimate_fix.py

# 步骤3: 提供选择
echo -e "\n步骤3: 选择修复方案..."
echo ""
echo "请选择修复方案:"
echo "1. 创建干净虚拟环境（推荐）"
echo "2. 修复当前环境"
echo "3. 使用无需bitsandbytes的版本"
echo "4. 运行详细诊断"
echo "5. 退出"
echo ""

read -p "请输入选择 (1-5): " CHOICE

case $CHOICE in
    1)
        echo -e "\n执行方案1: 创建干净虚拟环境"
        echo "----------------------------------------"
        
        # 检查Python3.10是否可用
        if command -v python3.10 &> /dev/null; then
            PYTHON_CMD="python3.10"
        elif command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        else
            PYTHON_CMD="python"
        fi
        
        VENV_PATH="$HOME/rlhf-fixed-env"
        
        echo "创建虚拟环境: $VENV_PATH"
        $PYTHON_CMD -m venv "$VENV_PATH"
        
        if [ $? -eq 0 ]; then
            echo "✅ 虚拟环境创建成功"
            echo ""
            echo "激活环境命令:"
            echo "  source $VENV_PATH/bin/activate"
            echo ""
            echo "安装依赖命令:"
            echo "  pip install torch==2.1.0 transformers==4.36.0 trl==0.11.0"
            echo "  pip install datasets==2.14.0 accelerate==0.25.0 peft==0.7.0"
            echo "  # 跳过bitsandbytes避免问题"
            echo ""
            echo "测试命令:"
            echo "  python -c \"from trl import PPOConfig; print('✅ 导入成功')\""
        else
            echo "❌ 虚拟环境创建失败"
        fi
        ;;
        
    2)
        echo -e "\n执行方案2: 修复当前环境"
        echo "----------------------------------------"
        
        echo "正在修复..."
        
        # 修复bitsandbytes问题
        echo "1. 处理bitsandbytes..."
        pip uninstall bitsandbytes -y 2>/dev/null || true
        echo "   bitsandbytes已卸载"
        
        # 修复transformers问题
        echo "2. 修复transformers..."
        pip install transformers==4.36.0 --force-reinstall
        
        # 修复TRL问题
        echo "3. 修复TRL..."
        pip install trl==0.11.0 --force-reinstall
        
        # 安装必要依赖
        echo "4. 安装必要依赖..."
        pip install datasets==2.14.0 accelerate==0.25.0 peft==0.7.0
        
        echo ""
        echo "✅ 修复完成"
        echo "测试命令: python minimal_test.py"
        ;;
        
    3)
        echo -e "\n执行方案3: 使用无需bitsandbytes的版本"
        echo "----------------------------------------"
        
        echo "运行无需bitsandbytes的测试..."
        python train_rlhf_no_bitsandbytes.py \
            --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
            --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
            --output_dir="./test_oneclick" \
            --max_samples=3
            
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 测试成功！"
            echo "可以使用以下命令进行完整训练:"
            echo "  python train_rlhf_no_bitsandbytes.py --max_samples=100"
        else
            echo ""
            echo "❌ 测试失败"
            echo "请尝试其他方案"
        fi
        ;;
        
    4)
        echo -e "\n执行方案4: 运行详细诊断"
        echo "----------------------------------------"
        
        echo "运行详细诊断..."
        python fix_triton_error.py
        python fix_import_issue.py
        python check_trl_structure.py
        
        echo ""
        echo "诊断完成，请根据输出解决问题"
        ;;
        
    5)
        echo "退出"
        exit 0
        ;;
        
    *)
        echo "无效选择"
        ;;
esac

# 步骤4: 总结
echo -e "\n=================================================="
echo "修复完成总结"
echo "=================================================="

echo "已创建的工具和脚本:"
echo "1. ultimate_fix.py        - 终极修复诊断"
echo "2. minimal_test.py        - 最小环境测试"
echo "3. emergency_solution.py  - 紧急解决方案"
echo "4. train_rlhf_no_bitsandbytes.py - 无需bitsandbytes版本"
echo "5. train_rlhf_universal.py       - 通用版本"
echo ""

echo "下一步建议:"
echo "1. 如果选择了方案1，先创建干净环境"
echo "2. 运行测试: python minimal_test.py"
echo "3. 验证模型加载: python simple_verify.py"
echo "4. 开始训练: python train_rlhf_no_bitsandbytes.py"
echo ""

echo "如果仍然有问题:"
echo "1. 查看详细错误信息"
echo "2. 运行: python emergency_solution.py"
echo "3. 联系系统管理员"
echo "4. 考虑使用Docker容器"

echo -e "\n所有修复脚本已就绪，祝你好运！"