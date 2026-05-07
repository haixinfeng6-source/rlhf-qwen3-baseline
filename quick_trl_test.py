#!/usr/bin/env python3
"""
快速TRL测试 - 找出正确的导入路径
"""

import sys

print("快速TRL导入测试")
print("=" * 50)

# 测试1: 直接导入
print("\n测试1: 直接导入")
try:
    from trl import PPOConfig
    print("✅ from trl import PPOConfig")
except ImportError as e:
    print(f"❌ from trl import PPOConfig: {e}")

# 测试2: 从trainer导入
print("\n测试2: 从trainer导入")
try:
    from trl.trainer import PPOConfig
    print("✅ from trl.trainer import PPOConfig")
except ImportError as e:
    print(f"❌ from trl.trainer import PPOConfig: {e}")

# 测试3: 检查TRL模块属性
print("\n测试3: 检查TRL模块属性")
try:
    import trl
    print(f"TRL模块属性: {[attr for attr in dir(trl) if not attr.startswith('_')][:10]}...")
    
    # 检查是否有trainer子模块
    if hasattr(trl, 'trainer'):
        print("✅ trl有trainer子模块")
        import trl.trainer
        print(f"trl.trainer属性: {[attr for attr in dir(trl.trainer) if not attr.startswith('_')][:10]}...")
    else:
        print("❌ trl没有trainer子模块")
        
except ImportError as e:
    print(f"❌ 导入trl失败: {e}")

# 测试4: 动态查找PPOConfig
print("\n测试4: 动态查找PPOConfig")
try:
    import trl
    
    # 在trl模块中查找
    if hasattr(trl, 'PPOConfig'):
        print(f"✅ 在trl中找到PPOConfig")
    
    # 在子模块中查找
    for attr_name in dir(trl):
        if not attr_name.startswith('_'):
            try:
                attr = getattr(trl, attr_name)
                if hasattr(attr, 'PPOConfig'):
                    print(f"✅ 在trl.{attr_name}中找到PPOConfig")
            except:
                pass
                
except Exception as e:
    print(f"❌ 动态查找失败: {e}")

# 测试5: 检查版本
print("\n测试5: 检查版本")
try:
    import importlib.metadata
    trl_version = importlib.metadata.version("trl")
    print(f"✅ TRL版本: {trl_version}")
    
    # 根据版本给出建议
    if trl_version.startswith('1.'):
        print("建议: TRL 1.x版本，尝试:")
        print("  from trl.trainer import PPOConfig, PPOTrainer")
        print("  from trl.models import AutoModelForCausalLMWithValueHead")
    elif trl_version.startswith('0.11.'):
        print("建议: TRL 0.11.x版本，尝试:")
        print("  from trl import PPOConfig, PPOTrainer")
        print("  from trl.models import AutoModelForCausalLMWithValueHead")
    elif trl_version.startswith('0.10.'):
        print("建议: TRL 0.10.x版本，尝试:")
        print("  from trl import PPOConfig, PPOTrainer")
        print("  from trl import AutoModelForCausalLMWithValueHead")
    else:
        print("未知版本，尝试所有可能")
        
except Exception as e:
    print(f"❌ 检查版本失败: {e}")

print("\n" + "=" * 50)
print("快速修复命令:")
print("=" * 50)
print("""
# 如果所有导入都失败，尝试:
pip uninstall trl -y
pip install trl==0.11.0

# 然后测试:
python -c "from trl import PPOConfig; print('✅ PPOConfig导入成功')"
python -c "from trl.models import AutoModelForCausalLMWithValueHead; print('✅ ValueHead导入成功')"

# 或者使用我们的通用脚本:
python train_rlhf_universal.py --max_samples=5
""")