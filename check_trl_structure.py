#!/usr/bin/env python3
"""
检查TRL包的结构，找出正确的导入路径
"""

import importlib
import importlib.metadata
import pkgutil
import sys

print("=" * 60)
print("TRL 包结构检查")
print("=" * 60)

# 检查TRL版本
try:
    trl_version = importlib.metadata.version("trl")
    print(f"TRL版本: {trl_version}")
except importlib.metadata.PackageNotFoundError:
    print("❌ TRL未安装")
    sys.exit(1)

# 检查TRL模块结构
print("\n检查TRL模块结构...")

def explore_module(module_name, depth=0, max_depth=2):
    """递归探索模块结构"""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print("  " * depth + f"✗ 无法导入 {module_name}: {e}")
        return
    
    prefix = "  " * depth
    print(f"{prefix}模块: {module_name}")
    
    # 列出模块的属性
    attrs = []
    for attr_name in dir(module):
        if not attr_name.startswith("_"):
            attrs.append(attr_name)
    
    # 显示前10个属性
    if attrs:
        print(f"{prefix}  属性: {', '.join(attrs[:10])}")
        if len(attrs) > 10:
            print(f"{prefix}  等... (共{len(attrs)}个属性)")
    
    # 如果是包，探索子模块
    if depth < max_depth and hasattr(module, "__path__"):
        try:
            for _, submodule_name, is_pkg in pkgutil.iter_modules(module.__path__):
                if depth == 0 and submodule_name in ["trainer", "models", "core"]:
                    explore_module(f"{module_name}.{submodule_name}", depth + 1, max_depth)
        except:
            pass

# 探索TRL主模块
explore_module("trl", depth=0, max_depth=2)

# 特别检查PPOConfig的位置
print("\n" + "=" * 60)
print("查找PPOConfig...")
print("=" * 60)

possible_paths = [
    "trl",
    "trl.trainer",
    "trl.trainer.ppo_config",
    "trl.core",
    "trl.models",
]

for path in possible_paths:
    try:
        module = importlib.import_module(path)
        if hasattr(module, "PPOConfig"):
            print(f"✅ 在 {path} 中找到 PPOConfig")
        else:
            # 检查子模块
            parts = path.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                child = ".".join(parts[i:])
                try:
                    parent_module = importlib.import_module(parent)
                    if hasattr(parent_module, child) and hasattr(getattr(parent_module, child), "PPOConfig"):
                        print(f"✅ 在 {parent}.{child} 中找到 PPOConfig")
                except:
                    pass
    except ImportError as e:
        print(f"✗ 无法导入 {path}: {e}")

# 检查PPOTrainer
print("\n" + "=" * 60)
print("查找PPOTrainer...")
print("=" * 60)

for path in possible_paths:
    try:
        module = importlib.import_module(path)
        if hasattr(module, "PPOTrainer"):
            print(f"✅ 在 {path} 中找到 PPOTrainer")
    except ImportError as e:
        print(f"✗ 无法导入 {path}: {e}")

# 检查AutoModelForCausalLMWithValueHead
print("\n" + "=" * 60)
print("查找AutoModelForCausalLMWithValueHead...")
print("=" * 60)

value_head_paths = [
    "trl",
    "trl.models",
    "trl.models.modeling_value_head",
    "trl.core",
]

for path in value_head_paths:
    try:
        module = importlib.import_module(path)
        if hasattr(module, "AutoModelForCausalLMWithValueHead"):
            print(f"✅ 在 {path} 中找到 AutoModelForCausalLMWithValueHead")
    except ImportError as e:
        print(f"✗ 无法导入 {path}: {e}")

# 提供导入建议
print("\n" + "=" * 60)
print("导入建议")
print("=" * 60)

print(f"根据TRL版本 {trl_version}，尝试以下导入:")

if trl_version.startswith("0.11."):
    print("""
# TRL 0.11.x 版本
from trl import PPOConfig, PPOTrainer
from trl.models import AutoModelForCausalLMWithValueHead
""")
elif trl_version.startswith("0.10."):
    print("""
# TRL 0.10.x 版本
from trl import PPOConfig, PPOTrainer
from trl import AutoModelForCausalLMWithValueHead
""")
elif trl_version.startswith("1."):
    print("""
# TRL 1.x 版本
try:
    from trl import PPOConfig, PPOTrainer
    from trl import AutoModelForCausalLMWithValueHead
except ImportError:
    # 可能在不同的子模块
    from trl.trainer import PPOConfig, PPOTrainer
    from trl.models import AutoModelForCausalLMWithValueHead
""")
else:
    print("""
# 未知版本，尝试所有可能
try:
    from trl import PPOConfig, PPOTrainer
    from trl.models import AutoModelForCausalLMWithValueHead
except ImportError:
    try:
        from trl.trainer import PPOConfig, PPOTrainer
        from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
    except ImportError:
        # 最后尝试
        import trl
        PPOConfig = trl.PPOConfig if hasattr(trl, 'PPOConfig') else None
        PPOTrainer = trl.PPOTrainer if hasattr(trl, 'PPOTrainer') else None
""")

# 测试实际导入
print("\n" + "=" * 60)
print("实际导入测试")
print("=" * 60)

test_imports = [
    ("from trl import PPOConfig, PPOTrainer", "标准导入"),
    ("from trl.trainer import PPOConfig, PPOTrainer", "trainer子模块导入"),
    ("from trl.models import AutoModelForCausalLMWithValueHead", "models导入ValueHead"),
    ("from trl import AutoModelForCausalLMWithValueHead", "直接导入ValueHead"),
]

for import_stmt, description in test_imports:
    try:
        exec(import_stmt)
        print(f"✅ {description}: 成功")
    except ImportError as e:
        print(f"✗ {description}: {e}")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)