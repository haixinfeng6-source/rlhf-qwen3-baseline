#!/usr/bin/env python3
"""
检查 TRL 1.3.0 的完整结构
"""

import sys
import inspect

print("=" * 60)
print("检查 TRL 1.3.0 完整结构")
print("=" * 60)

try:
    import trl
    print(f"TRL 版本: {getattr(trl, '__version__', '未知')}")
    
    # 检查 trl 的所有属性和子模块
    print("\ntrl 模块的所有属性:")
    for item in dir(trl):
        if not item.startswith('_'):
            obj = getattr(trl, item)
            if inspect.ismodule(obj):
                print(f"  📦 {item} (模块)")
            elif inspect.isclass(obj):
                print(f"  🏗️  {item} (类)")
            elif callable(obj):
                print(f"  🔧 {item} (函数)")
            else:
                print(f"  📄 {item}")
    
    # 特别检查是否有替代 AutoModelForCausalLMWithValueHead 的类
    print("\n搜索可能的替代类...")
    value_head_keywords = ['Value', 'value', 'Head', 'head', 'PPO', 'ppo', 'Model']
    for item in dir(trl):
        if any(keyword in item for keyword in value_head_keywords):
            obj = getattr(trl, item)
            if inspect.isclass(obj):
                print(f"  🔍 可能的相关类: {item}")
                # 显示类的文档
                doc = inspect.getdoc(obj)
                if doc:
                    print(f"     文档: {doc[:100]}...")
    
    # 检查子模块
    print("\n检查子模块...")
    submodules = ['models', 'trainer', 'core', 'imports', 'utils']
    for sub in submodules:
        try:
            module_name = f'trl.{sub}'
            module = __import__(module_name)
            print(f"\n📦 {module_name}:")
            # 获取实际模块
            for part in module_name.split('.')[1:]:
                module = getattr(module, part)
            
            # 列出前20个非私有属性
            items = [x for x in dir(module) if not x.startswith('_')]
            for i, item in enumerate(items[:20]):
                obj = getattr(module, item)
                if inspect.isclass(obj):
                    print(f"  🏗️  {item}")
                elif inspect.ismodule(obj):
                    print(f"  📦 {item}")
                elif callable(obj):
                    print(f"  🔧 {item}")
                else:
                    print(f"  📄 {item}")
            
            if len(items) > 20:
                print(f"  ... 还有 {len(items)-20} 个")
                
        except ImportError:
            print(f"  ✗ {module_name} 不存在")
        except Exception as e:
            print(f"  ✗ 检查 {module_name} 时出错: {e}")
    
    # 检查是否有 PPO 相关类
    print("\n搜索 PPO 相关类...")
    ppo_classes = []
    for item in dir(trl):
        if 'PPO' in item or 'Ppo' in item or 'ppo' in item:
            obj = getattr(trl, item)
            if inspect.isclass(obj):
                ppo_classes.append(item)
    
    if ppo_classes:
        print("找到的 PPO 相关类:")
        for cls in ppo_classes:
            print(f"  🏗️  {cls}")
    else:
        print("未找到 PPO 相关类")
        
except ImportError as e:
    print(f"✗ 无法导入 trl: {e}")
except Exception as e:
    print(f"✗ 检查失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)