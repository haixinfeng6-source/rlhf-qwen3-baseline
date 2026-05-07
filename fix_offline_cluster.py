"""
修复无网络集群环境问题
"""

import os
import sys
import subprocess


def check_and_install_deps():
    """检查并安装依赖包"""
    print("=" * 60)
    print("检查并安装依赖包")
    print("=" * 60)
    
    # 检查 wheels 目录是否存在
    wheels_dir = "./offline_assets/wheels"
    if not os.path.exists(wheels_dir):
        print(f"❌ 依赖包目录不存在: {wheels_dir}")
        print("请确保已传输 offline_assets 目录到集群")
        return False
    
    # 检查 wheel 文件数量
    wheel_files = [f for f in os.listdir(wheels_dir) if f.endswith('.whl')]
    print(f"找到 {len(wheel_files)} 个 wheel 文件")
    
    if len(wheel_files) == 0:
        print("❌ 未找到 wheel 文件")
        return False
    
    # 检查是否已安装核心依赖
    try:
        import torch
        print(f"✓ PyTorch 已安装: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch 未安装")
        print("开始安装依赖包...")
        
        # 安装依赖
        cmd = "pip install --no-index --find-links=./offline_assets/wheels -r requirements.txt"
        print(f"执行: {cmd}")
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ 依赖包安装成功")
            else:
                print(f"❌ 安装失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ 安装过程出错: {e}")
            return False
    
    return True


def check_offline_resources():
    """检查离线资源"""
    print("\n" + "=" * 60)
    print("检查离线资源")
    print("=" * 60)
    
    resources = [
        ("模型文件", "./offline_assets/models/Qwen--Qwen3-8B", True),
        ("数据集", "./offline_assets/datasets/openbmb--UltraFeedback", True),
        ("依赖包", "./offline_assets/wheels", True),
    ]
    
    all_ok = True
    for name, path, required in resources:
        if os.path.exists(path):
            if os.path.isdir(path):
                # 计算目录大小
                total_size = 0
                file_count = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                            file_count += 1
                
                size_gb = total_size / 1024**3
                print(f"✓ {name}: {path}")
                print(f"  文件数: {file_count}, 大小: {size_gb:.2f} GB")
            else:
                size_mb = os.path.getsize(path) / 1024**2
                print(f"✓ {name}: {path} ({size_mb:.2f} MB)")
        else:
            if required:
                print(f"❌ {name}: 路径不存在 {path}")
                all_ok = False
            else:
                print(f"⚠ {name}: 路径不存在 {path} (可选)")
    
    return all_ok


def create_offline_config():
    """创建离线配置文件"""
    print("\n" + "=" * 60)
    print("创建离线配置文件")
    print("=" * 60)
    
    config_content = """# 离线训练配置
# 适用于无网络集群

model:
  name: "./offline_assets/models/Qwen--Qwen3-8B"
  trust_remote_code: true

dataset:
  name: "./offline_assets/datasets/openbmb--UltraFeedback"
  split: "train"
  max_samples: null

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

quantization:
  use_8bit: true
  use_4bit: false

training:
  output_dir: "./output_offline"
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  max_grad_norm: 1.0
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  save_total_limit: 3
  max_length: 512
  gradient_checkpointing: true
  fp16: true

ppo:
  epochs: 4
  mini_batch_size: 1

distributed:
  num_gpus: 4

seed: 42
log_with: "tensorboard"
"""
    
    config_file = "config_offline.yaml"
    with open(config_file, "w") as f:
        f.write(config_content)
    
    print(f"✓ 创建离线配置文件: {config_file}")
    return True


def create_quick_start_script():
    """创建快速启动脚本"""
    print("\n" + "=" * 60)
    print("创建快速启动脚本")
    print("=" * 60)
    
    # Linux 脚本
    script_content = """#!/bin/bash

# 无网络集群快速启动脚本

echo "=========================================="
echo "RLHF 离线训练快速启动"
echo "=========================================="

# 检查环境
echo "1. 检查环境..."
python test_offline_cluster.py

if [ $? -ne 0 ]; then
    echo "❌ 环境检查失败，请先修复问题"
    exit 1
fi

echo ""
echo "2. 开始训练..."
echo "=========================================="

# 设置训练参数
MODEL_PATH="./offline_assets/models/Qwen--Qwen3-8B"
DATASET_PATH="./offline_assets/datasets/openbmb--UltraFeedback"
OUTPUT_DIR="./output_offline_$(date +%Y%m%d_%H%M%S)"

echo "模型: $MODEL_PATH"
echo "数据集: $DATASET_PATH"
echo "输出: $OUTPUT_DIR"
echo ""

# 询问是否继续
read -p "是否开始训练? (y/n): " CONTINUE
if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
    echo "已取消"
    exit 0
fi

# 开始训练
python train_rlhf.py \
    --model_name=$MODEL_PATH \
    --dataset_name=$DATASET_PATH \
    --output_dir=$OUTPUT_DIR \
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --use_lora \
    --lora_r=16 \
    --lora_alpha=32 \
    --use_8bit \
    --gradient_checkpointing \
    --logging_steps=10 \
    --save_steps=500 \
    --max_length=512

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: $OUTPUT_DIR"
echo ""
"""
    
    script_file = "quick_start_offline.sh"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_file, 0o755)
    
    print(f"✓ 创建快速启动脚本: {script_file}")
    print(f"  使用: bash {script_file}")
    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("无网络集群环境修复工具")
    print("=" * 60)
    print("此工具将帮助你在无网络集群上设置 RLHF 训练环境")
    print("=" * 60 + "\n")
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 执行修复步��
    steps = [
        ("检查离线资源", check_offline_resources),
        ("安装依赖包", check_and_install_deps),
        ("创建离线配置", create_offline_config),
        ("创建快速启动脚本", create_quick_start_script),
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n步骤: {step_name}")
        print("-" * 40)
        result = step_func()
        results.append((step_name, result))
    
    # 总结
    print("\n" + "=" * 60)
    print("修复完成")
    print("=" * 60)
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for step_name, result in results:
        status = "✓ 成功" if result else "❌ 失败"
        print(f"{step_name:.<40} {status}")
    
    print(f"\n成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✓ 所有修复步骤完成！")
        print("\n下一步:")
        print("1. 测试环境:")
        print("   python test_offline_cluster.py")
        print("\n2. 开始训练:")
        print("   bash quick_start_offline.sh")
        print("   或")
        print("   bash run_offline.sh")
        print("\n3. 多卡训练:")
        print("   bash run_offline_multi_gpu.sh")
    else:
        print("\n⚠ 部分修复步骤失败")
        print("\n请检查:")
        print("1. 确保 offline_assets 目录已正确传输")
        print("2. 检查磁盘空间是否足够")
        print("3. 参考 离线运行指南.md")
    
    print("=" * 60 + "\n")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
