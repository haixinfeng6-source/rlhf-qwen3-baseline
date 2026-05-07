#!/usr/bin/env python3
"""
测试版本兼容性修复
"""

import os
import sys

print("=" * 60)
print("测试版本兼容性修复")
print("=" * 60)

# 应用修复
print("1. 应用 huggingface_hub 修复...")
try:
    from huggingface_hub import is_offline_mode
    print("✓ huggingface_hub.is_offline_mode 已存在")
except ImportError:
    def is_offline_mode():
        return os.environ.get('HF_HUB_OFFLINE', '0') == '1'
    
    try:
        import huggingface_hub
        huggingface_hub.is_offline_mode = is_offline_mode
        print("✓ 为 huggingface_hub 添加了 is_offline_mode")
    except ImportError:
        print("⚠️  huggingface_hub 未安装")

print("\n2. 应用 pyarrow 修复...")
try:
    import pyarrow as pa
    if not hasattr(pa, 'PyExtensionType'):
        pa.PyExtensionType = pa.ExtensionType
        print("✓ 设置了 pa.PyExtensionType = pa.ExtensionType")
    else:
        print("✓ pyarrow.PyExtensionType 已存在")
except ImportError:
    print("⚠️  pyarrow 未安装")

print("\n3. 测试关键导入...")

# 测试 transformers
try:
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers: {e}")

# 测试 TRL
try:
    import trl
    print(f"✓ trl: {getattr(trl, '__version__', '未知')}")
    
    # 尝试导入 AutoModelForCausalLMWithValueHead
    try:
        from trl.models import AutoModelForCausalLMWithValueHead
        print("✓ AutoModelForCausalLMWithValueHead (from trl.models)")
    except ImportError:
        try:
            from trl import AutoModelForCausalLMWithValueHead
            print("✓ AutoModelForCausalLMWithValueHead (from trl)")
        except ImportError as e:
            print(f"✗ AutoModelForCausalLMWithValueHead: {e}")
            
except ImportError as e:
    print(f"✗ trl: {e}")

# 测试 PEFT
try:
    import peft
    print(f"✓ peft: {peft.__version__}")
except ImportError as e:
    print(f"✗ peft: {e}")

# 测试 datasets
try:
    import datasets
    print(f"✓ datasets: {datasets.__version__}")
except ImportError as e:
    print(f"✗ datasets: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

print("\n建议:")
print("1. 如果仍有导入错误，运行: bash fix_version_conflicts.sh")
print("2. 或者创建独立环境: bash install_rlhf_safe.sh")
print("3. 由于 CUDA 驱动太旧，训练将在 CPU 上进行")
print("4. 考虑更新 NVIDIA 驱动或使用云 GPU")