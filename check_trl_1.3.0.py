#!/usr/bin/env python3
"""
检查 TRL 1.3.0 版本的导入结构
"""

import sys
import importlib

print("=" * 60)
print("检查 TRL 1.3.0 导入结构")
print("=" * 60)

print(f"Python 版本: {sys.version}")

# 检查 TRL 版本
try:
    import trl
    print(f"TRL 版本: {trl.__version__}")
except AttributeError:
    import importlib.metadata
    trl_version = importlib.metadata.version("trl")
    print(f"TRL 版本: {trl_version}")

# 检查 trl 模块结构
print("\n检查 trl 模块结构...")
print("trl 目录:", [x for x in dir(trl) if not x.startswith('_')][:20])

# 检查 trl.models 模块
try:
    import trl.models
    print("\ntrl.models 目录:", [x for x in dir(trl.models) if not x.startswith('_')][:20])
    
    # 检查是否有 AutoModelForCausalLMWithValueHead
    if 'AutoModelForCausalLMWithValueHead' in dir(trl.models):
        print("✓ AutoModelForCausalLMWithValueHead 在 trl.models 中")
    else:
        print("✗ AutoModelForCausalLMWithValueHead 不在 trl.models 中")
except ImportError as e:
    print(f"✗ 无法导入 trl.models: {e}")

# 尝试不同的导入路径
print("\n尝试不同导入路径...")

# 1. 尝试从 trl 直接导入
try:
    from trl import AutoModelForCausalLMWithValueHead
    print("✓ 成功: from trl import AutoModelForCausalLMWithValueHead")
except ImportError as e:
    print(f"✗ 失败: from trl import AutoModelForCausalLMWithValueHead - {e}")

# 2. 尝试从 trl.models 导入
try:
    from trl.models import AutoModelForCausalLMWithValueHead
    print("✓ 成功: from trl.models import AutoModelForCausalLMWithValueHead")
except ImportError as e:
    print(f"✗ 失败: from trl.models import AutoModelForCausalLMWithValueHead - {e}")

# 3. 尝试从 trl.models.modeling 导入
try:
    from trl.models.modeling import AutoModelForCausalLMWithValueHead
    print("✓ 成功: from trl.models.modeling import AutoModelForCausalLMWithValueHead")
except ImportError as e:
    print(f"✗ 失败: from trl.models.modeling import AutoModelForCausalLMWithValueHead - {e}")

# 4. 尝试从 trl.models.modeling_value_head 导入
try:
    from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
    print("✓ 成功: from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead")
except ImportError as e:
    print(f"✗ 失败: from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead - {e}")

# 5. 检查所有可能的子模块
print("\n检查所有 trl 子模块...")
for submodule in ['models', 'models.modeling', 'models.modeling_value_head', 'core', 'trainer']:
    try:
        module = importlib.import_module(f'trl.{submodule}')
        print(f"✓ trl.{submodule} 存在")
        # 检查是否有 AutoModelForCausalLMWithValueHead
        if 'AutoModelForCausalLMWithValueHead' in dir(module):
            print(f"  ✓ AutoModelForCausalLMWithValueHead 在 trl.{submodule} 中")
    except ImportError:
        print(f"✗ trl.{submodule} 不存在")

print("\n" + "=" * 60)
print("建议:")
print("1. 检查 TRL 1.3.0 的文档或源码")
print("2. 尝试: from trl import AutoModelForCausalLMWithValueHead")
print("3. 如果仍然失败，可能需要降级到 TRL 0.11.0")
print("=" * 60)