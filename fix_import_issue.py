#!/usr/bin/env python3
"""
修复 TRL 和 Transformers 版本兼容性问题
解决 'is_torch_mlu_available' 导入错误
"""

import sys
import os

# 添加猴子补丁，在导入TRL之前修复
def patch_transformers_import():
    """猴子补丁修复 transformers.utils 导入问题"""
    import transformers.utils as utils_module
    
    # 检查是否已经有 is_torch_mlu_available
    if not hasattr(utils_module, 'is_torch_mlu_available'):
        # 添加一个假的 is_torch_mlu_available 函数
        def fake_is_torch_mlu_available():
            return False
        
        utils_module.is_torch_mlu_available = fake_is_torch_mlu_available
        print("✓ 已添加 is_torch_mlu_available 猴子补丁")
    
    # 同样处理其他可能缺失的导入
    if not hasattr(utils_module, 'is_torch_npu_available'):
        def fake_is_torch_npu_available():
            return False
        utils_module.is_torch_npu_available = fake_is_torch_npu_available
        print("✓ 已添加 is_torch_npu_available 猴子补丁")
    
    if not hasattr(utils_module, 'is_torch_xpu_available'):
        def fake_is_torch_xpu_available():
            return False
        utils_module.is_torch_xpu_available = fake_is_torch_xpu_available
        print("✓ 已添加 is_torch_xpu_available 猴子补丁")

def check_and_fix_environment():
    """检查并修复环境问题"""
    print("=" * 60)
    print("环境问题诊断和修复")
    print("=" * 60)
    
    # 1. 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 2. 检查PyTorch和CUDA
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠️ 警告: CUDA不可用")
        # 检查CUDA驱动版本
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"NVIDIA驱动版本: {result.stdout.strip()}")
                print("建议: 更新NVIDIA驱动到最新版本")
        except:
            pass
    
    # 3. 检查包版本
    import importlib.metadata
    packages = {
        'torch': '2.1.0+',
        'transformers': '4.36.0',
        'trl': '0.11.0',
        'datasets': '2.14.0',
        'accelerate': '0.25.0',
        'peft': '0.7.0'
    }
    
    print("\n包版本检查:")
    for pkg, recommended in packages.items():
        try:
            version = importlib.metadata.version(pkg)
            print(f"  {pkg:20} {version:15} ", end='')
            if pkg == 'torch':
                if '2.1' in version or '2.0' in version:
                    print("✓ 兼容")
                else:
                    print("⚠️ 建议使用2.1.x")
            elif pkg == 'transformers':
                if version == '4.36.0':
                    print("✓ 推荐版本")
                else:
                    print(f"⚠️ 建议使用{recommended}")
            elif pkg == 'trl':
                if version == '0.11.0':
                    print("✓ 推荐版本")
                else:
                    print(f"⚠️ 建议使用{recommended}")
            else:
                print("")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {pkg:20} ❌ 未安装")
    
    return True

def test_trl_import():
    """测试TRL导入是否正常"""
    print("\n" + "=" * 60)
    print("测试TRL导入")
    print("=" * 60)
    
    try:
        # 先应用猴子补丁
        patch_transformers_import()
        
        # 然后尝试导入TRL
        from trl import PPOConfig, PPOTrainer
        print("✓ PPOConfig, PPOTrainer 导入成功")
        
        # 测试AutoModelForCausalLMWithValueHead导入
        try:
            from trl.models import AutoModelForCausalLMWithValueHead
            print("✓ AutoModelForCausalLMWithValueHead (trl.models) 导入成功")
        except ImportError:
            try:
                from trl import AutoModelForCausalLMWithValueHead
                print("✓ AutoModelForCausalLMWithValueHead (trl) 导入成功")
            except ImportError as e:
                print(f"✗ AutoModelForCausalLMWithValueHead 导入失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ TRL导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 检查环境
    check_and_fix_environment()
    
    # 测试导入
    if test_trl_import():
        print("\n" + "=" * 60)
        print("✅ 环境修复成功！")
        print("=" * 60)
        print("\n建议修复步骤:")
        print("1. 创建干净环境:")
        print("   python3.10 -m venv ~/rlhf-fixed")
        print("   source ~/rlhf-fixed/bin/activate")
        print("")
        print("2. 安装兼容版本:")
        print("   pip install torch==2.1.0 transformers==4.36.0 trl==0.11.0")
        print("   pip install datasets==2.14.0 accelerate==0.25.0 peft==0.7.0")
        print("")
        print("3. 如果必须使用当前环境，运行:")
        print("   python -c \"exec(open('fix_import_issue.py').read())\"")
        print("   然后在代码开头添加: exec(open('fix_import_issue.py').read())")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ 环境修复失败")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())