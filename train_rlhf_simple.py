#!/usr/bin/env python3
"""
RLHF训练脚本 - 简化稳健版本
专为CPU-集群工作流程设计

特点:
1. 最小化依赖，避免兼容性问题
2. 不使用bitsandbytes，避免版本冲突
3. 支持CPU和GPU模式
4. 详细的错误处理和日志
5. 支持离线资源加载
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# 环境检查
# ============================================

def check_environment():
    """检查运行环境"""
    logger.info("=" * 60)
    logger.info("环境检查")
    logger.info("=" * 60)
    
    # Python版本
    logger.info(f"Python版本: {sys.version}")
    
    # PyTorch
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("CUDA不可用，将使用CPU模式")
            
        return True
        
    except ImportError as e:
        logger.error(f"PyTorch导入失败: {e}")
        return False

def check_dependencies():
    """检查关键依赖"""
    logger.info("\n检查依赖...")
    
    dependencies = {
        'transformers': '4.36.0',
        'trl': '0.11.0',
        'datasets': '2.14.0',
        'peft': '0.7.0',
    }
    
    all_ok = True
    
    for pkg, expected_version in dependencies.items():
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', '未知')
            logger.info(f"  {pkg:15} {version:10} ", end='')
            
            if version == expected_version:
                logger.info("✓")
            else:
                logger.warning(f"⚠ (期望 {expected_version})")
                
        except ImportError as e:
            logger.error(f"✗ 未安装")
            all_ok = False
            
    return all_ok

def check_trl_import():
    """检查TRL导入"""
    logger.info("\n检查TRL导入...")
    
    try:
        # 尝试导入关键类
        from trl import PPOConfig, PPOTrainer
        logger.info("  ✓ PPOConfig, PPOTrainer")
        
        from trl.models import AutoModelForCausalLMWithValueHead
        logger.info("  ✓ AutoModelForCausalLMWithValueHead")
        
        return True
        
    except ImportError as e:
        logger.error(f"  ✗ TRL导入失败: {e}")
        return False

# ============================================
# 配置加载
# ============================================

def load_config(config_path="config_cluster.yaml"):
    """加载配置文件"""
    logger.info(f"\n加载配置: {config_path}")
    
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}")
        logger.info("使用默认配置")
        return get_default_config()
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("✓ 配置加载成功")
        return config
        
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return get_default_config()

def get_default_config():
    """获取默认配置"""
    return {
        'model': {
            'name': './offline_assets/models/Qwen--Qwen3-8B',
            'trust_remote_code': True,
        },
        'dataset': {
            'name': './offline_assets/datasets/openbmb--UltraFeedback',
            'split': 'train',
            'max_samples': 100,
        },
        'lora': {
            'enabled': True,
            'r': 8,
            'alpha': 32,
            'dropout': 0.05,
        },
        'training': {
            'output_dir': './output_cluster',
            'num_train_epochs': 1,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 16,
            'learning_rate': 1e-5,
            'max_length': 256,
            'gradient_checkpointing': True,
            'fp16': True,
        },
    }

# ============================================
# 资源加载
# ============================================

def load_model_and_tokenizer(config):
    """加载模型和tokenizer"""
    logger.info("\n加载模型和tokenizer...")
    
    model_path = config['model']['name']
    logger.info(f"模型路径: {model_path}")
    
    # 检查路径
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return None, None
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config['model'].get('trust_remote_code', True),
            local_files_only=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"✓ Tokenizer加载成功 (词汇表: {len(tokenizer)})")
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        
        # 加载模型
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=config['model'].get('trust_remote_code', True),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True,
        )
        
        logger.info(f"✓ 模型加载成功")
        
        # LoRA
        if config['lora']['enabled']:
            logger.info("配置LoRA...")
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=config['lora']['r'],
                lora_alpha=config['lora']['alpha'],
                lora_dropout=config['lora']['dropout'],
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            logger.info("✓ LoRA配置成功")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_dataset(config, tokenizer):
    """加载数据集"""
    logger.info("\n加载数据集...")
    
    dataset_path = config['dataset']['name']
    logger.info(f"数据集路径: {dataset_path}")
    
    # 检查路径
    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return None
    
    try:
        from datasets import load_from_disk
        
        # 加载数据集
        dataset = load_from_disk(dataset_path)
        
        # 选择split
        split = config['dataset']['split']
        if hasattr(dataset, 'keys'):
            if split in dataset:
                dataset = dataset[split]
            else:
                dataset = dataset[list(dataset.keys())[0]]
        
        # 限制样本数
        max_samples = config['dataset'].get('max_samples')
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        logger.info(f"✓ 数据集加载成功 ({len(dataset)} 样本)")
        
        # 预处理
        logger.info("预处理数据集...")
        
        def preprocess(examples):
            queries = []
            for instruction in examples["instruction"]:
                messages = [{"role": "user", "content": instruction}]
                query = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                queries.append(query)
            return {"query": queries}
        
        dataset = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names,
            desc="预处理数据集",
        )
        
        logger.info("✓ 数据集预处理完成")
        
        return dataset
        
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# 主函数
# ============================================

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("RLHF训练 - 简化稳健版本")
    logger.info("=" * 60)
    
    # 1. 环境检查
    if not check_environment():
        logger.error("环境检查失败，请检查PyTorch安装")
        return 1
    
    if not check_dependencies():
        logger.error("依赖检查失败，请安装缺失的包")
        return 1
    
    if not check_trl_import():
        logger.error("TRL导入失败，请检查TRL安装")
        return 1
    
    # 2. 加载配置
    config = load_config()
    
    # 3. 加载模型
    model, tokenizer = load_model_and_tokenizer(config)
    if model is None or tokenizer is None:
        logger.error("模型加载失败")
        return 1
    
    # 4. 加载数据集
    dataset = load_dataset(config, tokenizer)
    if dataset is None:
        logger.error("数据集加载失败")
        return 1
    
    # 5. 总结
    logger.info("\n" + "=" * 60)
    logger.info("准备完成总结")
    logger.info("=" * 60)
    logger.info(f"模型: {config['model']['name']}")
    logger.info(f"数据集: {config['dataset']['name']} ({len(dataset)} 样本)")
    logger.info(f"输出目录: {config['training']['output_dir']}")
    logger.info(f"LoRA: {'启用' if config['lora']['enabled'] else '禁用'}")
    
    logger.info("\n✅ 所有准备工作完成！")
    logger.info("\n下一步:")
    logger.info("1. 配置PPO训练器")
    logger.info("2. 开始训练循环")
    logger.info("3. 保存模型检查点")
    
    # 创建输出目录
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_save_path = os.path.join(output_dir, "config_used.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"\n配置已保存: {config_save_path}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("\n用户中断")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"\n未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)