"""
使用配置文件的训练脚本
支持从 YAML 配置文件加载参数
"""

import os
import yaml
import torch
import argparse
from dataclasses import dataclass
from typing import Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: dict):
    """根据配置设置模型和 tokenizer"""
    model_config = config['model']
    lora_config = config['lora']
    quant_config = config['quantization']
    training_config = config['training']
    
    logger.info(f"Loading model: {model_config['name']}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        trust_remote_code=model_config.get('trust_remote_code', True),
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 量化配置
    quantization_config = None
    if quant_config.get('use_4bit'):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit quantization")
    elif quant_config.get('use_8bit'):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=model_config.get('trust_remote_code', True),
        torch_dtype=torch.float16 if not quantization_config else None,
    )
    
    # Gradient checkpointing
    if training_config.get('gradient_checkpointing'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # 准备量化训练
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA
    if lora_config.get('enabled'):
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            target_modules=lora_config['target_modules'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info("LoRA enabled")
    
    return model, tokenizer


def load_dataset_from_config(config: dict, tokenizer):
    """根据配置加载数据集"""
    dataset_config = config['dataset']
    
    logger.info(f"Loading dataset: {dataset_config['name']}")
    
    dataset = load_dataset(
        dataset_config['name'],
        split=dataset_config.get('split', 'train')
    )
    
    max_samples = dataset_config.get('max_samples')
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    def preprocess_function(examples):
        queries = []
        responses = []
        
        for instruction, completions in zip(examples["instruction"], examples["completions"]):
            messages = [{"role": "user", "content": instruction}]
            query = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            queries.append(query)
            
            best_completion = max(completions, key=lambda x: x.get("rating", 0))
            responses.append(best_completion["response"])
        
        return {"query": queries, "response": responses}
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset"
    )
    
    return dataset


def create_simple_reward_fn():
    """创建简单的奖励函数"""
    def reward_fn(samples):
        rewards = []
        for sample in samples:
            # 基于长度和完整性的简单奖励
            # 实际应用中应使用训练好的 reward model
            length_reward = min(len(sample.split()) / 100, 1.0)
            
            # 检查是否有完整的句子结构
            completeness_reward = 0.5 if sample.strip().endswith(('.', '!', '?')) else 0.0
            
            total_reward = length_reward + completeness_reward
            rewards.append(total_reward)
        
        return rewards
    
    return reward_fn


def train_ppo(config: dict, model, tokenizer, dataset):
    """PPO 训练"""
    training_config = config['training']
    ppo_config_dict = config['ppo']
    
    # 包装为 PPO 模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    # PPO 配置
    ppo_config = PPOConfig(
        model_name=config['model']['name'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['per_device_train_batch_size'],
        mini_batch_size=ppo_config_dict['mini_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        ppo_epochs=ppo_config_dict['epochs'],
        max_grad_norm=training_config['max_grad_norm'],
        seed=config.get('seed', 42),
        log_with=config.get('log_with', 'tensorboard'),
        project_kwargs={"logging_dir": os.path.join(training_config['output_dir'], "logs")},
    )
    
    # 创建 trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # 奖励函数
    reward_fn = create_simple_reward_fn()
    
    logger.info("Starting PPO training...")
    
    # 训练循环
    for epoch in range(training_config['num_train_epochs']):
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            
            # 生成响应
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_length=training_config['max_length'],
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
            
            # 解码
            batch["response"] = tokenizer.batch_decode(
                response_tensors,
                skip_special_tokens=True
            )
            
            # 计算奖励
            rewards = reward_fn(batch["response"])
            rewards = [torch.tensor(r) for r in rewards]
            
            # PPO 更新
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # 日志
            if batch_idx % training_config['logging_steps'] == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: {stats}")
            
            # 保存
            if batch_idx % training_config['save_steps'] == 0 and batch_idx > 0:
                save_path = os.path.join(
                    training_config['output_dir'],
                    f"checkpoint-{epoch}-{batch_idx}"
                )
                ppo_trainer.save_pretrained(save_path)
                logger.info(f"Checkpoint saved to {save_path}")
    
    # 保存最终模型
    final_path = os.path.join(training_config['output_dir'], "final_model")
    ppo_trainer.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training completed! Model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="RLHF Training with Config File")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ppo", "grpo"],
        default=None,
        help="Training method (overrides config)"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # 命令行参数覆盖配置
    if args.method:
        config['training_method'] = args.method
    
    # 设置随机种子
    torch.manual_seed(config.get('seed', 42))
    
    # 加载模型和数据
    model, tokenizer = setup_model_and_tokenizer(config)
    dataset = load_dataset_from_config(config, tokenizer)
    
    # 根据方法选择训练
    method = config.get('training_method', 'ppo')
    
    if method == 'ppo':
        train_ppo(config, model, tokenizer, dataset)
    elif method == 'grpo':
        logger.error("GRPO training not implemented in this script yet")
        logger.info("Please use train_grpo.py for GRPO training")
    else:
        raise ValueError(f"Unknown training method: {method}")


if __name__ == "__main__":
    main()
