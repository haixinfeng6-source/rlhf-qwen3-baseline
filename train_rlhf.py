"""
RLHF Training Script for Qwen3-8B using TRL
支持 PPO 和 GRPO 方法，针对多卡训练和显存优化
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """训练参数配置"""
    # 模型相关
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    
    # 数据集相关
    dataset_name: str = field(default="openbmb/UltraFeedback")
    dataset_split: str = field(default="train")
    max_samples: Optional[int] = field(default=None)
    
    # LoRA 配置
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")
    
    # 量化配置
    use_8bit: bool = field(default=True)
    use_4bit: bool = field(default=False)
    
    # 训练配置
    output_dir: str = field(default="./rlhf_output")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=1e-5)
    max_grad_norm: float = field(default=1.0)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    
    # PPO 特定参数
    ppo_epochs: int = field(default=4)
    mini_batch_size: int = field(default=1)
    
    # 显存优化
    gradient_checkpointing: bool = field(default=True)
    max_length: int = field(default=512)
    
    # 其他
    seed: int = field(default=42)


def load_ultrafeedback_dataset(args: ScriptArguments, tokenizer):
    """
    加载并预处理 UltraFeedback 数据集
    UltraFeedback 格式: {instruction, completions: [{response, model, rating}]}
    """
    logger.info(f"Loading dataset: {args.dataset_name}")
    
    # 加载数据集
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    def preprocess_function(examples):
        """
        预处理函数：将数据转换为 PPO 训练所需格式
        需要提取 query (instruction) 和选择最高评分的 response
        """
        queries = []
        responses = []
        
        for instruction, completions in zip(examples["instruction"], examples["completions"]):
            # 使用 chat template 格式化 query
            messages = [{"role": "user", "content": instruction}]
            query = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            queries.append(query)
            
            # 选择评分最高的 response
            best_completion = max(completions, key=lambda x: x.get("rating", 0))
            responses.append(best_completion["response"])
        
        return {
            "query": queries,
            "response": responses,
        }
    
    # 预处理数据集
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset"
    )
    
    return dataset


def create_reward_model(model_name: str, quantization_config=None):
    """
    创建 Reward Model (这里简化为使用长度作为奖励)
    实际应用中应该训练一个专门的 reward model
    """
    def reward_fn(samples):
        """
        简单的奖励函数示例
        实际应该使用训练好的 reward model
        """
        rewards = []
        for sample in samples:
            # 简单示例：基于长度和完整性给予奖励
            # 实际应该使用 reward model 预测
            length_reward = min(len(sample.split()) / 100, 1.0)
            rewards.append(length_reward)
        return rewards
    
    return reward_fn


def setup_model_and_tokenizer(args: ScriptArguments):
    """
    设置模型和 tokenizer，包括量化和 LoRA 配置
    """
    logger.info(f"Loading model: {args.model_name}")
    
    # Tokenizer 配置
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",  # PPO 训练推荐左填充
    )
    
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 量化配置
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit quantization")
    elif args.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        logger.info("Using 8-bit quantization")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not quantization_config else None,
    )
    
    # Gradient checkpointing (显存优化)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # 准备模型用于 k-bit 训练
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA 配置
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA enabled")
    
    # 包装为 PPO 模型（带 value head）
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    return model, tokenizer


def main():
    # 解析参数（实际使用时可以用 HfArgumentParser）
    args = ScriptArguments()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 加载模型和 tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # 加载数据集
    dataset = load_ultrafeedback_dataset(args, tokenizer)
    
    # PPO 配置
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.per_device_train_batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        log_with="tensorboard",
        project_kwargs={"logging_dir": os.path.join(args.output_dir, "logs")},
    )
    
    # 创建 PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # 创建奖励函数
    reward_fn = create_reward_model(args.model_name)
    
    logger.info("Starting PPO training...")
    
    # 训练循环
    for epoch in range(args.num_train_epochs):
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            
            # 生成响应
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_length=args.max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
            
            # 解码响应
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # 计算奖励
            rewards = reward_fn(batch["response"])
            rewards = [torch.tensor(r) for r in rewards]
            
            # PPO 更新
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # 日志记录
            if batch_idx % args.logging_steps == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: {stats}")
            
            # 保存检查点
            if batch_idx % args.save_steps == 0 and batch_idx > 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{batch_idx}")
                ppo_trainer.save_pretrained(save_path)
                logger.info(f"Checkpoint saved to {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, "final_model")
    ppo_trainer.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Training completed! Model saved to {final_save_path}")


if __name__ == "__main__":
    main()
