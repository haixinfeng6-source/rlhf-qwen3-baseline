#!/usr/bin/env python3
"""
CPU 版本的 RLHF 训练脚本
用于 CUDA 不可用或驱动太旧的环境
"""

import os
import sys

# 强制使用 CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['LOCAL_RANK'] = '-1'

print("=" * 60)
print("CPU 版本 RLHF 训练")
print("注意: 由于 CUDA 不可用，使用 CPU 进行训练")
print("训练速度会很慢，仅用于测试和调试")
print("=" * 60)

# 修改 train_rlhf.py 的导入和配置
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print("警告: CUDA 可用，但使用 CPU 模式")
    torch.cuda.is_available = lambda: False  # 临时覆盖

# 导入修改后的训练逻辑
try:
    # 导入必要的模块
    from dataclasses import dataclass, field
    from typing import Optional
    
    @dataclass
    class TrainingArguments:
        """简化的训练参数"""
        model_name: str = "Qwen/Qwen3-8B"
        dataset_name: str = "openbmb/UltraFeedback"
        output_dir: str = "./output_cpu_test"
        max_samples: Optional[int] = 5  # 只测试5个样本
        use_lora: bool = False  # CPU 上禁用 LoRA
        use_8bit: bool = False  # CPU 上禁用 8-bit
        use_4bit: bool = False  # CPU 上禁用 4-bit
        per_device_train_batch_size: int = 1
        gradient_accumulation_steps: int = 1
        num_train_epochs: int = 1
        learning_rate: float = 1e-5
        max_length: int = 128  # 缩短序列长度
        seed: int = 42
    
    print("✓ 参数类定义成功")
    
    # 测试导入
    print("\n测试导入...")
    
    # TRL 导入
    try:
        from trl.models import AutoModelForCausalLMWithValueHead
        print("✓ AutoModelForCausalLMWithValueHead 导入成功 (TRL 0.11.0+)")
    except ImportError:
        try:
            from trl import AutoModelForCausalLMWithValueHead
            print("✓ AutoModelForCausalLMWithValueHead 导入成功 (TRL 0.10.0)")
        except ImportError as e:
            print(f"✗ AutoModelForCausalLMWithValueHead 导入失败: {e}")
            sys.exit(1)
    
    from trl import PPOConfig, PPOTrainer
    print("✓ PPOConfig, PPOTrainer 导入成功")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✓ Transformers 导入成功")
    
    from datasets import load_dataset
    print("✓ Datasets 导入成功")
    
    print("\n" + "=" * 60)
    print("所有导入测试通过!")
    print("=" * 60)
    
    # 创建简单的测试训练
    print("\n开始简单测试训练...")
    
    # 创建输出目录
    os.makedirs("./output_cpu_test", exist_ok=True)
    
    # 加载一个小模型进行测试（不使用 Qwen，因为它太大）
    print("加载小型测试模型...")
    try:
        # 尝试加载小模型
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # 包装为 PPO 模型
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        
        print("✓ 模型加载成功")
        
        # 创建虚拟数据
        print("创建虚拟训练数据...")
        class DummyDataset:
            def __init__(self):
                self.data = [
                    {"query": "What is AI?", "response": "AI is artificial intelligence."},
                    {"query": "What is ML?", "response": "ML is machine learning."},
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = DummyDataset()
        
        # PPO 配置
        ppo_config = PPOConfig(
            model_name="gpt2",
            learning_rate=1e-5,
            batch_size=1,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
        )
        
        # 创建 PPO Trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
        )
        
        print("✓ PPO Trainer 创建成功")
        print("\n" + "=" * 60)
        print("CPU 测试训练准备完成!")
        print("=" * 60)
        print("\n说明:")
        print("1. 由于 CUDA 不可用，使用 CPU 模式")
        print("2. 使用小型模型 gpt2 进行测试")
        print("3. 使用虚拟数据进行测试")
        print("4. 实际训练需要修复 CUDA 驱动或使用 GPU 服务器")
        
    except Exception as e:
        print(f"✗ 测试训练失败: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ 脚本初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)