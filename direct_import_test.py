#!/usr/bin/env python3
"""
直接导入测试 - 不运行训练，只测试导入
"""

import os
import sys

print("=" * 60)
print("直接导入测试")
print("=" * 60)

# 设置环境变量
os.environ['LOCAL_RANK'] = '-1'

# 测试 1: 检查 Python 和 PyTorch
print("1. 检查基础环境...")
try:
    import torch
    print(f"  ✓ PyTorch 版本: {torch.__version__}")
    print(f"  ✓ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU 数量: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"  ✗ PyTorch 导入失败: {e}")
    sys.exit(1)

# 测试 2: 检查 TRL 导入
print("\n2. 检查 TRL 导入...")
try:
    # 使用 train_rlhf.py 中的导入逻辑
    import importlib.metadata
    
    trl_version = importlib.metadata.version("trl")
    print(f"  ✓ TRL 版本: {trl_version}")
    
    # 尝试导入 AutoModelForCausalLMWithValueHead
    AutoModelForCausalLMWithValueHead = None
    
    # 尝试不同的导入路径
    import_paths = [
        ("trl", "直接导入"),
        ("trl.models", "models 模块"),
        ("trl.models.modeling_value_head", "modeling_value_head"),
        ("trl.models.modeling", "modeling 模块"),
    ]
    
    for import_path, description in import_paths:
        try:
            if '.' in import_path:
                module_name, class_name = import_path.rsplit('.', 1)
                module = __import__(module_name, fromlist=[class_name])
                AutoModelForCausalLMWithValueHead = getattr(module, 'AutoModelForCausalLMWithValueHead')
            else:
                module = __import__(import_path)
                AutoModelForCausalLMWithValueHead = getattr(module, 'AutoModelForCausalLMWithValueHead')
            
            print(f"  ✓ {description} 成功: {import_path}")
            break
        except (ImportError, AttributeError) as e:
            print(f"    ✗ {description} 失败: {e}")
            continue
    
    if AutoModelForCausalLMWithValueHead is None:
        print("  ✗ 所有导入路径都失败!")
    else:
        print(f"  ✓ 成功获取 AutoModelForCausalLMWithValueHead: {AutoModelForCausalLMWithValueHead}")
        
except Exception as e:
    print(f"  ✗ TRL 检查失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 3: 检查其他关键导入
print("\n3. 检查其他关键导入...")
imports_to_check = [
    ("transformers", ["AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig"]),
    ("peft", ["LoraConfig", "get_peft_model", "prepare_model_for_kbit_training"]),
    ("datasets", ["load_dataset"]),
    ("trl", ["PPOConfig", "PPOTrainer"]),
]

for module_name, attributes in imports_to_check:
    try:
        module = __import__(module_name)
        print(f"  ✓ {module_name} 导入成功")
        
        for attr in attributes:
            try:
                getattr(module, attr)
                print(f"    ✓ {attr} 可用")
            except AttributeError:
                # 特殊处理 trl 的子模块
                if module_name == "trl" and attr in ["PPOConfig", "PPOTrainer"]:
                    try:
                        from trl.trainer import PPOConfig, PPOTrainer
                        print(f"    ✓ {attr} 可用 (从 trl.trainer 导入)")
                    except ImportError:
                        print(f"    ✗ {attr} 不可用")
                else:
                    print(f"    ✗ {attr} 不可用")
                    
    except ImportError as e:
        print(f"  ✗ {module_name} 导入失败: {e}")

print("\n" + "=" * 60)
print("导入测试完成")
print("=" * 60)