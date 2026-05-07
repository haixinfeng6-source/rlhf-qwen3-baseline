#!/usr/bin/env python3
"""
测试离线资源设置
验证离线模型和数据集是否可以正常加载
"""

import os
import sys

print("=" * 60)
print("离线资源测试")
print("=" * 60)

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 检查离线资源目录
base_path = "offline_assets"
print(f"检查目录: {base_path}")

# 检查模型
model_path = os.path.join(base_path, "models", "Qwen--Qwen3-8B")
print(f"\n1. 检查模型: {model_path}")

if os.path.exists(model_path):
    print("   ✓ 模型目录存在")
    
    # 检查关键文件
    required_files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024 / 1024
            print(f"   ✓ {file} ({size:.1f} MB)")
        else:
            # 检查 safetensors 文件
            if file == "model.safetensors.index.json":
                # 查找实际的 safetensors 文件
                safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
                if safetensors_files:
                    print(f"   ✓ 找到 safetensors 文件: {len(safetensors_files)} 个")
                else:
                    print(f"   ✗ {file} 不存在")
else:
    print("   ✗ 模型目录不存在")

# 检查数据集
dataset_path = os.path.join(base_path, "datasets", "openbmb--UltraFeedback")
print(f"\n2. 检查数据集: {dataset_path}")

if os.path.exists(dataset_path):
    print("   ✓ 数据集目录存在")
    
    # 尝试加载数据集
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        
        # 检查是否是 DatasetDict
        if hasattr(dataset, 'keys'):
            print(f"   ✓ 加载成功: DatasetDict with keys: {list(dataset.keys())}")
            # 获取 train 分割
            if 'train' in dataset:
                train_dataset = dataset['train']
                print(f"   ✓ Train 分割: {len(train_dataset)} 样本")
        else:
            print(f"   ✓ 加载成功: {len(dataset)} 样本")
            
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")
else:
    print("   ✗ 数据集目录不存在")

# 检查 wheels
wheels_path = os.path.join(base_path, "wheels")
print(f"\n3. 检查 Python 包: {wheels_path}")

if os.path.exists(wheels_path):
    print("   ✓ Wheels 目录存在")
    
    # 统计 wheel 文件
    wheel_files = [f for f in os.listdir(wheels_path) if f.endswith('.whl')]
    print(f"   ✓ 找到 {len(wheel_files)} 个 wheel 文件")
    
    # 检查关键包
    key_packages = ['torch', 'transformers', 'trl', 'datasets', 'peft']
    for pkg in key_packages:
        pkg_files = [f for f in wheel_files if f.startswith(pkg)]
        if pkg_files:
            print(f"   ✓ {pkg}: {len(pkg_files)} 个版本")
        else:
            print(f"   ⚠️  {pkg}: 未找到")
else:
    print("   ✗ Wheels 目录不存在")

# 测试导入
print("\n4. 测试导入 (离线模式):")

import_tests = [
    ("import torch", "PyTorch"),
    ("from transformers import AutoTokenizer, AutoModelForCausalLM", "Transformers"),
    ("from datasets import load_from_disk", "Datasets"),
]

for import_stmt, description in import_tests:
    try:
        exec(import_stmt)
        print(f"   ✓ {description}")
    except ImportError as e:
        print(f"   ✗ {description}: {e}")

# 测试模型加载
print("\n5. 测试模型加载:")

if os.path.exists(model_path):
    try:
        from transformers import AutoTokenizer
        
        # 测试 tokenizer 加载
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("   ✓ Tokenizer 加载成功")
        print(f"     词汇表大小: {len(tokenizer)}")
        
    except Exception as e:
        print(f"   ✗ Tokenizer 加载失败: {e}")
else:
    print("   ⚠️  跳过模型加载测试")

print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

# 给出建议
issues = []

if not os.path.exists(model_path):
    issues.append("❌ 模型目录不存在")
if not os.path.exists(dataset_path):
    issues.append("❌ 数据集目录不存在")
if not os.path.exists(wheels_path):
    issues.append("❌ Wheels 目录不存在")

if issues:
    print("发现的问题:")
    for issue in issues:
        print(f"  {issue}")
    
    print("\n解决方案:")
    print("  1. 确保 offline_assets/ 目录结构正确")
    print("  2. 检查文件权限")
    print("  3. 如果需要，重新下载资源")
else:
    print("✅ 所有离线资源检查通过")
    
    print("\n下一步:")
    print("  1. 安装依赖: pip install --no-index --find-links=offline_assets/wheels -r requirements.txt")
    print("  2. 测试训练: python train_rlhf.py --model_name='./offline_assets/models/Qwen--Qwen3-8B' --dataset_name='./offline_assets/datasets/openbmb--UltraFeedback' --max_samples=5")
    print("  3. 完整训练: bash run_offline.sh")

print("\n" + "=" * 60)