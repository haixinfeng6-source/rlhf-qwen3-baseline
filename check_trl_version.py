#!/usr/bin/env python3
"""
检查 TRL 版本和导入路径
"""

import sys
import importlib.metadata

print("=" * 60)
print("TRL 版本检查")
print("=" * 60)

# 检查 Python 版本
print(f"Python 版本: {sys.version}")

# 检查 TRL 版本
try:
    trl_version = importlib.metadata.version("trl")
    print(f"TRL 版本: {trl_version}")
except importlib.metadata.PackageNotFoundError:
    print("TRL 未安装")
    sys.exit(1)

# 检查导入路径
print("\n检查导入路径...")

# 检查 trl.models 模块
try:
    import trl.models
    print("✓ trl.models 模块存在")
    
    # 列出 trl.models 中的内容
    print(f"  trl.models 内容: {dir(trl.models)[:10]}...")
    
    # 检查 AutoModelForCausalLMWithValueHead
    if 'AutoModelForCausalLMWithValueHead' in dir(trl.models):
        print("✓ AutoModelForCausalLMWithValueHead 在 trl.models 中")
    else:
        print("✗ AutoModelForCausalLMWithValueHead 不在 trl.models 中")
except ImportError as e:
    print(f"✗ trl.models 模块不存在: {e}")

# 检查 trl 模块
try:
    import trl
    print("\n✓ trl 模块存在")
    
    # 列出 trl 中的内容
    print(f"  trl 内容: {[x for x in dir(trl) if not x.startswith('_')][:15]}...")
    
    # 检查 AutoModelForCausalLMWithValueHead
    if 'AutoModelForCausalLMWithValueHead' in dir(trl):
        print("✓ AutoModelForCausalLMWithValueHead 在 trl 中")
    else:
        print("✗ AutoModelForCausalLMWithValueHead 不在 trl 中")
except ImportError as e:
    print(f"✗ trl 模块不存在: {e}")

# 尝试导入
print("\n尝试导入...")
try:
    from trl.models import AutoModelForCausalLMWithValueHead
    print("✓ 成功: from trl.models import AutoModelForCausalLMWithValueHead")
except ImportError as e:
    print(f"✗ 失败: from trl.models import AutoModelForCausalLMWithValueHead - {e}")

try:
    from trl import AutoModelForCausalLMWithValueHead
    print("✓ 成功: from trl import AutoModelForCausalLMWithValueHead")
except ImportError as e:
    print(f"✗ 失败: from trl import AutoModelForCausalLMWithValueHead - {e}")

# 检查其他关键导入
print("\n检查其他导入...")
try:
    from trl import PPOConfig, PPOTrainer
    print("✓ PPOConfig, PPOTrainer 导入成功")
except ImportError as e:
    print(f"✗ PPOConfig, PPOTrainer 导入失败: {e}")

print("\n" + "=" * 60)
print("建议:")
print("1. 如果 TRL 版本 < 0.11.0，请升级: pip install trl>=0.11.0")
print("2. 如果使用 Python 3.13，建议降级到 Python 3.10 或 3.11")
print("3. 检查 requirements.txt 中的版本要求")
print("=" * 60)