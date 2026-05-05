"""
RLHF Training Script for Qwen using TRL
基于 HuggingFace TRL 官方 PPO 实现的标准 RLHF pipeline
参考: https://github.com/huggingface/trl
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import load_dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    """RLHF 训练参数配置"""
    
    # 模型配置
    model_name: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "预训练模型名称或路径"}
    )
    
    # 数据集配置
    dataset_name: str = field(
        default="openbmb/UltraFeedback",
        metadata={"help": "数据集名称"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "数据集分割"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "最大样本数，None 表示使用全部"}
    )
    
    # LoRA 配置
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用 LoRA"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    
    # 量化配置
    use_8bit: bool = field(
        default=True,
        metadata={"help": "使用 8-bit 量化"}
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "使用 4-bit 量化"}
    )
    
    # 训练配置
    output_dir: str = field(
        default="./output_rlhf",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "训练轮数"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "每个设备的批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "梯度累积步数"}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "学习率"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "梯度裁剪"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "预热步数"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志记录步数"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "保存步数"}
    )
    
    # PPO 特定参数
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "PPO 内部迭代次数"}
    )
    mini_batch_size: int = field(
        default=1,
        metadata={"help": "PPO mini-batch 大小"}
    )
    
    # 显存优化
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "启用 gradient checkpointing"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    
    # 其他
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )


def load_and_preprocess_dataset(args: TrainingArguments, tokenizer):
    """
    加载并预处理 UltraFeedback 数据集
    支持从本地路径或 HuggingFace Hub 加载
    UltraFeedback 格式: {instruction, completions: [{response, model, rating}]}
    """
    logger.info(f"加载数据集: {args.dataset_name}")
    
    # 判断是本地路径还是 HuggingFace 数据集名称
    if os.path.exists(args.dataset_name):
        # 从本地加载（离线模式）
        from datasets import load_from_disk
        logger.info("从本地路径加载数据集（离线模式）")
        dataset = load_from_disk(args.dataset_name)
        # 如果是 DatasetDict，获取指定的 split
        if hasattr(dataset, 'keys'):
            if args.dataset_split in dataset:
                dataset = dataset[args.dataset_split]
            else:
                # 如果没有指定的 split，使用第一个
                dataset = dataset[list(dataset.keys())[0]]
                logger.warning(f"未找到 split '{args.dataset_split}'，使用 '{list(dataset.keys())[0]}'")
    else:
        # 从 HuggingFace Hub 加载（在线模式）
        logger.info("从 HuggingFace Hub 加载数据集（在线模式）")
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    def preprocess_function(examples):
        """
        预处理函数：将数据转换为 PPO 训练所需格式
        提取 query (instruction) 和选择最高评分的 response
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
        desc="预处理数据集"
    )
    
    return dataset


def create_reward_function():
    """
    创建奖励函数
    
    注意: 这是一个简化的示例奖励函数
    实际应用中应该使用训练好的 Reward Model
    """
    def reward_fn(samples: list) -> list:
        """
        计算奖励
        
        Args:
            samples: 生成的文本列表
            
        Returns:
            rewards: 奖励值列表
        """
        rewards = []
        for sample in samples:
            # 简单示例：基于长度和完整性给予奖励
            # 实际应该使用 reward model 预测
            
            # 长度奖励 (鼓励适当长度的回答)
            length = len(sample.split())
            length_reward = min(length / 100, 1.0)
            
            # 完整性奖励 (鼓励完整的句子)
            completeness_reward = 0.5 if sample.strip().endswith(('.', '!', '?')) else 0.0
            
            # 总奖励
            total_reward = length_reward + completeness_reward
            rewards.append(total_reward)
        
        return rewards
    
    return reward_fn


def setup_model_and_tokenizer(args: TrainingArguments):
    """
    设置模型和 tokenizer，包括量化和 LoRA 配置
    """
    logger.info(f"加载模型: {args.model_name}")
    
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
        logger.info("使用 4-bit 量化")
    elif args.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        logger.info("使用 8-bit 量化")
    
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
        logger.info("Gradient checkpointing 已启用")
    
    # 准备模型用于 k-bit 训练
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA 配置
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA 已启用")
    
    # 包装为 PPO 模型（带 value head）
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    return model, tokenizer


def main():
    """主训练函数"""
    
    # 解析参数
    from transformers import HfArgumentParser
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和 tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # 加载数据集
    dataset = load_and_preprocess_dataset(args, tokenizer)
    
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
    reward_fn = create_reward_function()
    
    logger.info("=" * 60)
    logger.info("开始 RLHF (PPO) 训练")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model_name}")
    logger.info(f"数据集: {args.dataset_name} ({len(dataset)} 样本)")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"训练轮数: {args.num_train_epochs}")
    logger.info(f"批次大小: {args.per_device_train_batch_size}")
    logger.info(f"梯度累积: {args.gradient_accumulation_steps}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info("=" * 60)
    
    # 训练循环
    for epoch in range(args.num_train_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        logger.info(f"{'='*60}")
        
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
                logger.info(f"检查点已保存到 {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, "final_model")
    ppo_trainer.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    logger.info("=" * 60)
    logger.info(f"训练完成！模型已保存到 {final_save_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
