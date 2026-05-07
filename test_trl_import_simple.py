#!/usr/bin/env python3
"""
简单的 TRL 导入测试
"""

import sys

print("测试 TRL 导入...")

# 方法1: 直接尝试导入
try:
    from trl import AutoModelForCausalLMWithValueHead
    print("✓ 方法1成功: from trl import AutoModelForCausalLMWithValueHead")
except ImportError as e:
    print(f"✗ 方法1失败: {e}")

# 方法2: 使用 importlib 动态导入
try:
    import importlib
    trl_module = importlib.import_module('trl')
    
    # 检查 trl 模块中有什么
    print("\ntrl 模块内容 (前20个):")
    items = [x for x in dir(trl_module) if not x.startswith('_')]
    for i, item in enumerate(items[:20]):
        print(f"  {i+1:2d}. {item}")
    
    # 检查是否有 AutoModelForCausalLMWithValueHead
    if 'AutoModelForCausalLMWithValueHead' in items:
        print("\n✓ AutoModelForCausalLMWithValueHead 在 trl 模块中")
        AutoModelForCausalLMWithValueHead = getattr(trl_module, 'AutoModelForCausalLMWithValueHead')
        print(f"✓ 成功获取 AutoModelForCausalLMWithValueHead: {AutoModelForCausalLMWithValueHead}")
    else:
        print("\n✗ AutoModelForCausalLMWithValueHead 不在 trl 模块中")
        
except Exception as e:
    print(f"✗ 方法2失败: {e}")

# 方法3: 检查所有子模块
print("\n检查子模块...")
submodules = ['models', 'models.modeling', 'models.modeling_value_head', 'core', 'trainer', 'imports']
for sub in submodules:
    try:
        module = importlib.import_module(f'trl.{sub}')
        items = [x for x in dir(module) if not x.startswith('_')]
        if 'AutoModelForCausalLMWithValueHead' in items:
            print(f"✓ 在 trl.{sub} 中找到 AutoModelForCausalLMWithValueHead")
            break
    except ImportError:
        continue
    except Exception as e:
        print(f"  检查 trl.{sub} 时出错: {e}")

print("\n" + "=" * 60)
print("如果所有方法都失败，建议:")
print("1. 降级到 TRL 0.11.0: pip install trl==0.11.0")
print("2. 或者检查 TRL 1.3.0 的文档")
print("=" * 60)