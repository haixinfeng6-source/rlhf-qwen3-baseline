#!/usr/bin/env python3
"""
集群环境测试脚本
用于验证RLHF训练环境是否配置正确
"""

import os
import sys
import torch
import importlib.metadata

print("=" * 60)
print("RLHF 集群环境测试")
print("=" * 60)

# 1. 检查Python版本
print("1. Python版本检查:")
print(f"   Python: {sys.version}")
print(f"   版本: {sys.version_info}")

# 2. 检查PyTorch
print("\n2. PyTorch检查:")
print(f"   版本: {torch.__version__}")
print(f"   CUDA可用: {torch.cuda.is_available()}")
print(f"   CUDA版本: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"   GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"     GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"       内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("   ⚠️ 警告: CUDA不可用，无法进行GPU训练")

# 3. 检查分布式环境
print("\n3. 分布式环境检查:")
local_rank = os.environ.get("LOCAL_RANK", "未设置")
world_size = os.environ.get("WORLD_SIZE", "未设置")
print(f"   LOCAL_RANK: {local_rank}")
print(f"   WORLD_SIZE: {world_size}")
print(f"   MASTER_ADDR: {os.environ.get('MASTER_ADDR', '未设置')}")
print(f"   MASTER_PORT: {os.environ.get('MASTER_PORT', '未设置')}")

# 4. 检查关键包版本
print("\n4. 包版本检查:")
packages = [
    "torch", "transformers", "trl", "datasets", 
    "accelerate", "peft", "bitsandbytes"
]

for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"   {pkg:20} {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"   {pkg:20} ❌ 未安装")

# 5. 检查TRL特定导入
print("\n5. TRL导入检查:")
try:
    # 检查TRL版本
    trl_version = importlib.metadata.version("trl")
    print(f"   TRL版本: {trl_version}")
    
    # 尝试导入关键类
    if trl_version.startswith('1.'):
        print("   TRL 1.x版本检测")
        try:
            from trl import AutoModelForCausalLMWithValueHead
            print("   ✓ AutoModelForCausalLMWithValueHead (trl)")
        except ImportError:
            try:
                from trl.models import AutoModelForCausalLMWithValueHead
                print("   ✓ AutoModelForCausalLMWithValueHead (trl.models)")
            except ImportError as e:
                print(f"   ❌ 导入失败: {e}")
    else:
        print("   TRL 0.x版本检测")
        try:
            from trl.models import AutoModelForCausalLMWithValueHead
            print("   ✓ AutoModelForCausalLMWithValueHead (trl.models)")
        except ImportError:
            try:
                from trl import AutoModelForCausalLMWithValueHead
                print("   ✓ AutoModelForCausalLMWithValueHead (trl)")
            except ImportError as e:
                print(f"   ❌ 导入失败: {e}")
    
    # 检查PPO相关导入
    try:
        from trl import PPOConfig, PPOTrainer
        print("   ✓ PPOConfig, PPOTrainer")
    except ImportError as e:
        print(f"   ❌ PPO导入失败: {e}")
        
except importlib.metadata.PackageNotFoundError:
    print("   ❌ TRL未安装")

# 6. 检查离线资源
print("\n6. 离线资源检查:")
offline_paths = [
    ("模型", "./offline_assets/models/Qwen--Qwen3-8B"),
    ("数据集", "./offline_assets/datasets/openbmb--UltraFeedback"),
]

for name, path in offline_paths:
    if os.path.exists(path):
        size = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                size += os.path.getsize(os.path.join(root, file))
        print(f"   {name:10} ✓ {path}")
        print(f"             大小: {size / 1024**3:.2f} GB")
    else:
        print(f"   {name:10} ❌ {path} (不存在)")

# 7. 测试简单模型加载
print("\n7. 模型加载测试:")
try:
    from transformers import AutoTokenizer
    
    # 测试tokenizer加载
    test_model_path = "./offline_assets/models/Qwen--Qwen3-8B"
    if os.path.exists(test_model_path):
        print(f"   测试加载tokenizer: {test_model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                test_model_path,
                trust_remote_code=True
            )
            print(f"   ✓ Tokenizer加载成功")
            print(f"     词汇表大小: {len(tokenizer)}")
            print(f"     pad_token: {tokenizer.pad_token}")
        except Exception as e:
            print(f"   ❌ Tokenizer加载失败: {e}")
    else:
        print("   ⚠️ 模型路径不存在，跳过加载测试")
except ImportError:
    print("   ⚠️ Transformers未安装，跳过加载测试")

# 8. 环境变量检查
print("\n8. 环境变量检查:")
env_vars = [
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
    "NCCL_DEBUG", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES"
]

for var in env_vars:
    value = os.environ.get(var, "未设置")
    print(f"   {var:25} {value}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

# 总结
print("\n总结:")
issues = []

if not torch.cuda.is_available():
    issues.append("CUDA不可用 - 无法进行GPU训练")

try:
    trl_version = importlib.metadata.version("trl")
    if trl_version.startswith('1.'):
        issues.append(f"TRL版本为{trl_version}，建议使用0.11.0以避免API变化")
except:
    issues.append("TRL未安装")

if not os.path.exists("./offline_assets/models/Qwen--Qwen3-8B"):
    issues.append("模型文件不存在")

if not os.path.exists("./offline_assets/datasets/openbmb--UltraFeedback"):
    issues.append("数据集文件不存在")

if issues:
    print("⚠️ 发现以下问题:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print(f"\n共发现 {len(issues)} 个问题需要解决")
else:
    print("✅ 所有检查通过，环境配置正确")

print("\n建议:")
print("1. 如果TRL版本为1.x，运行: pip install trl==0.11.0")
print("2. 确保离线资源路径正确")
print("3. 在提交作业前，先运行单卡测试")
print("4. 使用提供的修复脚本: train_rlhf_cluster_fixed.py")

print("\n单卡测试命令:")
print("python train_rlhf_cluster_fixed.py \\")
print("  --model_name=\"./offline_assets/models/Qwen--Qwen3-8B\" \\")
print("  --dataset_name=\"./offline_assets/datasets/openbmb--UltraFeedback\" \\")
print("  --output_dir=\"./test_output\" \\")
print("  --max_samples=10")

print("\n" + "=" * 60)