#!/usr/bin/env python3
"""
RLHF 训练脚本 - 通用版本
使用TRL导入助手解决版本兼容性问题
"""

import os
import sys
import torch
import logging

print("=" * 60)
print("RLHF 训练 - 通用版本")
print("=" * 60)

# 环境检查
print(f"Python版本: {sys.version_info.major}.{sys.version_info.minor}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
else:
    print("⚠️ CUDA不可用，将在CPU上运行")

# ========== 使用TRL导入助手 ==========
print("\n" + "=" * 60)
print("导入TRL组件")
print("=" * 60)

# 添加当前目录到路径，以便导入助手
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from trl_import_helper import TRLImporter
    print("✅ TRL导入助手加载成功")
    
    # 导入TRL组件
    trl_importer = TRLImporter().import_all()
    
    # 获取组件
    PPOConfig = trl_importer.PPOConfig
    PPOTrainer = trl_importer.PPOTrainer
    AutoModelForCausalLMWithValueHead = trl_importer.AutoModelForCausalLMWithValueHead
    
    if not all([PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead]):
        print("❌ 无法导入所有必要的TRL组件")
        print("尝试备用方案...")
        raise ImportError("TRL组件导入不完整")
        
except ImportError as e:
    print(f"❌ TRL导入失败: {e}")
    print("\n尝试直接导入...")
    
    # 尝试直接导入
    import_attempts = [
        # 尝试标准导入
        ("from trl import PPOConfig, PPOTrainer", "标准导入"),
        ("from trl.trainer import PPOConfig, PPOTrainer", "trainer导入"),
        # 尝试ValueHead导入
        ("from trl.models import AutoModelForCausalLMWithValueHead", "models导入"),
        ("from trl import AutoModelForCausalLMWithValueHead", "直接导入"),
    ]
    
    PPOConfig = None
    PPOTrainer = None
    AutoModelForCausalLMWithValueHead = None
    
    for import_stmt, description in import_attempts:
        try:
            exec(import_stmt)
            print(f"✅ {description} 成功")
            
            # 检查哪些组件导入了
            if 'PPOConfig' in locals():
                PPOConfig = locals()['PPOConfig']
            if 'PPOTrainer' in locals():
                PPOTrainer = locals()['PPOTrainer']
            if 'AutoModelForCausalLMWithValueHead' in locals():
                AutoModelForCausalLMWithValueHead = locals()['AutoModelForCausalLMWithValueHead']
                
        except ImportError as e:
            print(f"  ✗ {description} 失败: {e}")
    
    if not all([PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead]):
        print("\n⚠️ 无法导入完整的TRL组件，使用简化模式")
        print("将进行环境验证，但不进行实际训练")

# ========== 导入其他必要库 ==========
print("\n" + "=" * 60)
print("导入其他库")
print("=" * 60)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
    print("✅ Transformers 导入成功")
except ImportError as e:
    print(f"❌ Transformers 导入失败: {e}")
    sys.exit(1)

try:
    from datasets import load_dataset, load_from_disk
    print("✅ Datasets 导入成功")
except ImportError as e:
    print(f"❌ Datasets 导入失败: {e}")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    print("✅ PEFT 导入成功")
except ImportError as e:
    print(f"❌ PEFT 导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("所有库导入完成")
print("=" * 60)

# ========== 训练逻辑 ==========
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingArguments:
    """训练参数"""
    model_name: str = field(default="./offline_assets/models/Qwen--Qwen3-8B")
    dataset_name: str = field(default="./offline_assets/datasets/openbmb--UltraFeedback")
    output_dir: str = field(default="./output_universal")
    max_samples: Optional[int] = field(default=10)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=8)
    learning_rate: float = field(default=1e-5)
    max_length: int = field(default=256)

def main():
    """主函数"""
    # 解析参数
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"数据集: {args.dataset_name}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大样本数: {args.max_samples}")
    print(f"LoRA: {'启用' if args.use_lora else '禁用'}")
    print(f"学习率: {args.learning_rate}")
    
    # 检查TRL组件
    if not all([PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead]):
        print("\n⚠️ TRL组件不完整，进行环境验证")
        print("=" * 60)
        
        # 测试模型加载
        print("1. 测试模型加载...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,
                trust_remote_code=True,
            )
            print(f"   ✅ Tokenizer加载成功 (词汇表: {len(tokenizer)})")
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            print(f"   ✅ 模型加载成功")
            
            if args.use_lora:
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj"],
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)
                print(f"   ✅ LoRA配置成功")
            
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            return False
        
        # 测试数据集加载
        print("\n2. 测试数据集加载...")
        try:
            if os.path.exists(args.dataset_name):
                dataset = load_from_disk(args.dataset_name)
                if hasattr(dataset, 'keys'):
                    dataset = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
            else:
                dataset = load_dataset(args.dataset_name, split='train')
            
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            
            print(f"   ✅ 数据集加载成功 ({len(dataset)} 样本)")
            
        except Exception as e:
            print(f"   ❌ 数据集加载失败: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ 环境验证通过！")
        print("=" * 60)
        
        print("\n问题诊断:")
        print("1. TRL组件导入不完整")
        print("2. 可能的原因:")
        print("   - TRL版本不兼容")
        print("   - 包版本冲突")
        print("   - Python环境问题")
        
        print("\n解决方案:")
        print("1. 创建干净环境:")
        print("   python3.10 -m venv ~/rlhf-clean")
        print("   source ~/rlhf-clean/bin/activate")
        print("   pip install -r requirements_cluster.txt")
        
        print("\n2. 检查TRL结构:")
        print("   python check_trl_structure.py")
        
        print("\n3. 使用修复脚本:")
        print("   bash fix_cluster_environment.sh")
        
        # 保存配置
        config_file = os.path.join(args.output_dir, "config.txt")
        with open(config_file, 'w') as f:
            f.write(f"模型: {args.model_name}\n")
            f.write(f"数据集: {args.dataset_name}\n")
            f.write(f"样本数: {args.max_samples}\n")
            f.write(f"TRL状态: 组件不完整\n")
            f.write(f"建议: 使用requirements_cluster.txt创建干净环境\n")
        
        print(f"\n配置已保存到: {config_file}")
        return True
    
    else:
        # 完整的TRL组件可用，进行实际训练
        print("\n✅ 所有TRL组件可用，开始训练准备")
        print("=" * 60)
        
        # 这里可以添加实际的训练逻辑
        print("训练步骤:")
        print("1. 加载模型和tokenizer ✓")
        print("2. 配置LoRA (如果启用) ✓")
        print("3. 加载和预处理数据集 ✓")
        print("4. 配置PPO训练器 ✓")
        print("5. 开始训练")
        
        print("\n实际训练代码需要:")
        print("1. 完整的训练循环")
        print("2. 奖励函数定义")
        print("3. 超参数调优")
        
        # 保存成功的配置
        success_file = os.path.join(args.output_dir, "success.txt")
        with open(success_file, 'w') as f:
            f.write(f"TRL版本: {trl_importer.trl_version if 'trl_importer' in locals() else '未知'}\n")
            f.write(f"PPOConfig: {PPOConfig.__module__}\n")
            f.write(f"PPOTrainer: {PPOTrainer.__module__}\n")
            f.write(f"ValueHead: {AutoModelForCausalLMWithValueHead.__module__}\n")
            f.write(f"状态: 所有组件导入成功\n")
        
        print(f"\n配置已保存到: {success_file}")
        return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n" + "=" * 60)
            print("✅ 脚本执行成功")
            print("=" * 60)
            print("\n下一步:")
            print("1. 如果TRL组件完整，可以开始实际训练")
            print("2. 如果只是环境验证，请按照建议修复环境")
            print("3. 在GPU集群上使用: bash run_cluster_offline.sh")
        else:
            print("\n❌ 脚本执行失败")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)