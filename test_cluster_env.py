#!/usr/bin/env python3
"""
集群环境测试脚本
验证环境配置是否正确
"""

import os
import sys
import platform

print("=" * 60)
print("集群环境测试")
print("=" * 60)

# 系统信息
print(f"系统: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"工作目录: {os.getcwd()}")

# 环境变量
print("\n环境变量检查:")
env_vars = [
    'LOCAL_RANK', 'WORLD_SIZE', 'RANK',
    'CUDA_VISIBLE_DEVICES', 'NCCL_DEBUG',
    'HF_HOME', 'TRANSFORMERS_CACHE'
]

for var in env_vars:
    value = os.environ.get(var, '未设置')
    print(f"  {var}: {value}")

# PyTorch 检查
print("\nPyTorch 检查:")
try:
    import torch
    print(f"  ✓ PyTorch 版本: {torch.__version__}")
    print(f"  ✓ CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  ✓ GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️  CUDA 不可用，确保在 GPU 节点运行")
        
except ImportError as e:
    print(f"  ✗ PyTorch 导入失败: {e}")

# 关键包检查
print("\n关键包检查:")
packages = [
    ('transformers', '4.36.0'),
    ('trl', '0.11.0'),
    ('datasets', '2.14.0'),
    ('peft', '0.7.0'),
    ('accelerate', '0.25.0'),
]

for pkg, expected_version in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', '未知')
        status = "✓" if version.startswith(expected_version) else "⚠️"
        print(f"  {status} {pkg}: {version}")
    except ImportError as e:
        print(f"  ✗ {pkg}: 未安装 ({e})")

# 导入测试
print("\n导入测试:")
import_tests = [
    ("from transformers import AutoModelForCausalLM, AutoTokenizer", "Transformers"),
    ("from trl.models import AutoModelForCausalLMWithValueHead", "TRL 模型"),
    ("from trl import PPOConfig, PPOTrainer", "TRL 训练器"),
    ("from datasets import load_dataset", "Datasets"),
    ("from peft import LoraConfig", "PEFT"),
]

for import_stmt, description in import_tests:
    try:
        exec(import_stmt)
        print(f"  ✓ {description}")
    except ImportError as e:
        print(f"  ✗ {description}: {e}")

# 文件系统检查
print("\n文件系统检查:")
paths_to_check = [
    '.',
    'offline_assets/models',
    'offline_assets/datasets',
    'offline_assets/wheels',
]

for path in paths_to_check:
    if os.path.exists(path):
        if os.path.isdir(path):
            # 统计文件数量
            try:
                file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print(f"  ✓ {path}/ (目录, {file_count} 个文件)")
            except:
                print(f"  ✓ {path}/ (目录)")
        else:
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  ✓ {path} ({size:.1f} MB)")
    else:
        print(f"  ⚠️  {path}: 不存在")

# 多进程测试
print("\n多进程环境测试:")
local_rank = os.environ.get('LOCAL_RANK', '-1')
world_size = os.environ.get('WORLD_SIZE', '1')

if local_rank != '-1':
    print(f"  ✓ 检测到多进程环境")
    print(f"    LOCAL_RANK: {local_rank}")
    print(f"    WORLD_SIZE: {world_size}")
else:
    print("  ⚠️  单进程环境")

print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

# 给出建议
print("\n建议:")
if not torch.cuda.is_available():
    print("1. ❌ CUDA 不可用 - 确保在 GPU 节点运行")
else:
    print("1. ✅ CUDA 可用 - 可以开始训练")

missing_packages = []
for pkg, _ in packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print(f"2. ❌ 缺少包: {', '.join(missing_packages)}")
    print(f"   安装命令: pip install {' '.join(missing_packages)}")
else:
    print("2. ✅ 所有必要包已安装")

print("3. 下一步:")
print("   - 单卡测试: python train_rlhf_cluster.py")
print("   - 多卡测试: bash run_multi_gpu.sh")
print("   - 集群作业: sbatch submit_job.slurm")

print("\n" + "=" * 60)