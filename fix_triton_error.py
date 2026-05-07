#!/usr/bin/env python3
"""
修复 'ModuleNotFoundError: No module named 'triton.ops'' 错误
bitsandbytes 库的 Triton 依赖问题
"""

import os
import sys
import subprocess
import importlib

print("=" * 60)
print("修复 Triton.ops 模块缺失错误")
print("=" * 60)

def check_triton():
    """检查Triton安装状态"""
    print("1. 检查Triton安装...")
    
    try:
        import triton
        print(f"   ✅ Triton版本: {triton.__version__}")
        
        # 检查ops子模块
        try:
            import triton.ops
            print("   ✅ triton.ops 模块可用")
            return True
        except ImportError as e:
            print(f"   ❌ triton.ops 导入失败: {e}")
            return False
            
    except ImportError:
        print("   ❌ Triton未安装")
        return False

def check_bitsandbytes():
    """检查bitsandbytes安装状态"""
    print("\n2. 检查bitsandbytes安装...")
    
    try:
        import bitsandbytes
        print(f"   ✅ bitsandbytes版本: {bitsandbytes.__version__}")
        
        # 测试bitsandbytes功能
        try:
            import bitsandbytes.nn
            print("   ✅ bitsandbytes.nn 模块可用")
            return True
        except ImportError as e:
            print(f"   ❌ bitsandbytes.nn 导入失败: {e}")
            return False
            
    except ImportError:
        print("   ❌ bitsandbytes未安装")
        return False

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    print("\n3. 检查PyTorch CUDA支持...")
    
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("   ⚠️ CUDA不可用，bitsandbytes可能无法正常工作")
            return False
            
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def diagnose_issue():
    """诊断问题根源"""
    print("\n" + "=" * 60)
    print("问题诊断")
    print("=" * 60)
    
    triton_ok = check_triton()
    bnb_ok = check_bitsandbytes()
    cuda_ok = check_pytorch_cuda()
    
    print("\n诊断结果:")
    
    if not triton_ok:
        print("❌ 问题: Triton未安装或损坏")
        print("   原因: bitsandbytes需要Triton进行优化")
        
    if not bnb_ok:
        print("❌ 问题: bitsandbytes未安装或损坏")
        
    if not cuda_ok:
        print("⚠️ 警告: CUDA不可用")
        print("   bitsandbytes需要CUDA支持")
    
    return triton_ok, bnb_ok, cuda_ok

def provide_solutions(triton_ok, bnb_ok, cuda_ok):
    """提供解决方案"""
    print("\n" + "=" * 60)
    print("解决方案")
    print("=" * 60)
    
    solutions = []
    
    # 方案1: 重新安装bitsandbytes（如果CUDA可用）
    if cuda_ok:
        solutions.append("""
方案1: 重新安装bitsandbytes（推荐）
----------------------------------------
# 卸载现有版本
pip uninstall bitsandbytes -y

# 安装兼容版本（根据CUDA版本）
# CUDA 11.8:
pip install bitsandbytes==0.41.0

# 或从源码安装（最可靠）:
pip install git+https://github.com/TimDettmers/bitsandbytes.git
""")
    else:
        solutions.append("""
方案1: 安装CPU版本的bitsandbytes
----------------------------------------
# 如果CUDA不可用，安装CPU版本
pip uninstall bitsandbytes -y
pip install bitsandbytes-cpu

# 或者在代码中禁用bitsandbytes:
# 设置环境变量: export BITSANDBYTES_NOWELCOME=1
# 或禁用8-bit量化: --use_8bit=False
""")
    
    # 方案2: 安装/修复Triton
    if not triton_ok:
        solutions.append("""
方案2: 安装Triton
----------------------------------------
# 安装Triton（PyTorch的编译器）
pip install triton==2.0.0

# 或者安装与PyTorch兼容的版本:
# PyTorch 2.0+: pip install triton==2.0.0
# PyTorch 2.1+: pip install triton==2.1.0
""")
    
    # 方案3: 使用替代方案
    solutions.append("""
方案3: 使用替代方案（无需bitsandbytes）
----------------------------------------
# 方法A: 禁用8-bit量化
# 在训练命令中添加: --use_8bit=False

# 方法B: 使用4-bit量化（如果支持）
# 添加: --use_4bit=True

# 方法C: 使用普通FP16训练
# 添加: --use_8bit=False --use_4bit=False

# 方法D: 使用LoRA但不量化
# 添加: --use_lora --use_8bit=False
""")
    
    # 方案4: 创建干净环境
    solutions.append("""
方案4: 创建干净环境（彻底解决）
----------------------------------------
# 1. 创建新环境
python3.10 -m venv ~/rlhf-clean-bnb
source ~/rlhf-clean-bnb/bin/activate

# 2. 安装基础包
pip install torch==2.1.0 transformers==4.36.0

# 3. 安装bitsandbytes（可选）
# 如果CUDA可用:
pip install bitsandbytes==0.41.0
# 如果CUDA不可用:
pip install bitsandbytes-cpu

# 4. 安装其他依赖
pip install trl==0.11.0 datasets==2.14.0 accelerate==0.25.0
""")
    
    # 打印所有方案
    for i, solution in enumerate(solutions, 1):
        print(f"\n{solution}")
    
    return solutions

def create_workaround_script():
    """创建临时解决方案脚本"""
    script_content = '''#!/usr/bin/env python3
"""
临时解决方案：绕过bitsandbytes错误
在bitsandbytes不可用时使用替代方案
"""

import os
import sys

def patch_bitsandbytes_import():
    """猴子补丁修复bitsandbytes导入"""
    
    # 方法1: 设置环境变量禁用bitsandbytes
    os.environ['BITSANDBYTES_NOWELCOME'] = '1'
    os.environ['DISABLE_BITSANDBYTES'] = '1'
    
    # 方法2: 创建假的bitsandbytes模块
    import types
    
    class FakeBitsAndBytesConfig:
        """假的BitsAndBytesConfig类"""
        def __init__(self, **kwargs):
            self.load_in_8bit = kwargs.get('load_in_8bit', False)
            self.load_in_4bit = kwargs.get('load_in_4bit', False)
            print("⚠️ 使用假的BitsAndBytesConfig（bitsandbytes不可用）")
    
    # 将假类添加到sys.modules
    fake_module = types.ModuleType('bitsandbytes')
    fake_module.BitsAndBytesConfig = FakeBitsAndBytesConfig
    sys.modules['bitsandbytes'] = fake_module
    
    # 也添加到transformers的导入路径
    sys.modules['transformers.utils.bitsandbytes'] = fake_module
    
    print("✅ 已应用bitsandbytes猴子补丁")

def check_environment():
    """检查环境并建议配置"""
    import torch
    
    print("环境检查:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠️ 警告: CUDA不可用，建议:")
        print("  1. 使用 --use_8bit=False")
        print("  2. 减小批次大小")
        print("  3. 使用梯度检查点")
    
    # 检查内存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i} 内存: {mem:.1f} GB")
            
            if mem < 16:
                print(f"  ⚠️ GPU {i} 内存不足，建议禁用8-bit量化")

def get_recommended_args():
    """获取推荐的训练参数"""
    args = {
        'use_8bit': False,      # 禁用8-bit量化
        'use_4bit': False,      # 禁用4-bit量化
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'gradient_checkpointing': True,
        'fp16': True,           # 使用FP16而不是8-bit
    }
    
    return args

if __name__ == "__main__":
    print("=" * 60)
    print("bitsandbytes错误临时解决方案")
    print("=" * 60)
    
    # 应用补丁
    patch_bitsandbytes_import()
    
    # 检查环境
    check_environment()
    
    # 获取推荐参数
    args = get_recommended_args()
    
    print("\n推荐训练参数:")
    for key, value in args.items():
        print(f"  --{key}={value}")
    
    print("\n使用示例:")
    print("python train_rlhf.py \\")
    print("  --model_name='./offline_assets/models/Qwen--Qwen3-8B' \\")
    print("  --use_8bit=False \\")
    print("  --gradient_checkpointing \\")
    print("  --per_device_train_batch_size=1")
    
    print("\n" + "=" * 60)
    print("临时解决方案已就绪")
    print("=" * 60)
'''
    
    script_path = "workaround_bitsandbytes.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n✅ 已创建临时解决方案脚本: {script_path}")
    print(f"使用方法: python {script_path}")
    
    return script_path

def main():
    """主函数"""
    # 诊断问题
    triton_ok, bnb_ok, cuda_ok = diagnose_issue()
    
    # 提供解决方案
    solutions = provide_solutions(triton_ok, bnb_ok, cuda_ok)
    
    # 创建临时解决方案
    script_path = create_workaround_script()
    
    print("\n" + "=" * 60)
    print("立即行动建议")
    print("=" * 60)
    
    print("""
步骤1: 尝试最简单的解决方案
----------------------------------------
# 禁用8-bit量化
python train_rlhf_universal.py --use_8bit=False --max_samples=2

步骤2: 如果步骤1失败，使用临时解决方案
----------------------------------------
python workaround_bitsandbytes.py
# 然后按照输出建议的参数运行

步骤3: 如果还需要bitsandbytes，重新安装
----------------------------------------
# 根据CUDA情况选择
pip uninstall bitsandbytes -y

# CUDA可用:
pip install bitsandbytes==0.41.0

# CUDA不可用:
pip install bitsandbytes-cpu

步骤4: 创建干净环境（终极方案）
----------------------------------------
# 按照方案4创建全新环境
""")
    
    print(f"\n详细解决方案已保存到: {script_path}")
    print("运行 'python workaround_bitsandbytes.py' 查看具体建议")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())