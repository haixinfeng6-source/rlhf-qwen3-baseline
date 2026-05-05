"""
GRPO (Group Relative Policy Optimization) Training Script
GRPO 是一种更简单高效的 RLHF 方法，不需要 value model
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
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """训练参数配置"""
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    dataset_name: str = field(default="openbmb/UltraFeedback")
    dataset_split: str = field(default="train")
    max_samples: Optional[int] = field(default=None)
    
    # LoRA 配置
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    
    # 量化
    use_8bit: bool = field(default=True)
    
    # 训练配置
    output_dir: str = field(default="./grpo_output")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-6)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    
    # GRPO 特定参数
    num_generations: int = field(default=4)  # 每个 query 生成多个响应
    temperature: float = field(default=0.8)
    
    # 显存优化
    gradient_checkpointing: bool = field(default=True)
    max_length: int = field(default=512)
    
    seed: int = field(default=42)


def load_and_preprocess_dataset(args: ScriptArguments, tokenizer):
    """加载并预处理 UltraFeedback 数据集用于 GRPO"""
    logger.info(f"Loading dataset: {args.dataset_name}")
    
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    def preprocess_function(examples):
        """
        GRPO 需要: query + multiple responses with scores
        """
        processed = {
            "query": [],
            "chosen": [],
            "rejected": [],
        }
        
        for instruction, completions in zip(examples["instruction"], examples["completions"]):
            if len(completions) < 2:
                continue
            
            # 格式化 query
            messages = [{"role": "user", "content": instruction}]
            query = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 按评分排序
            sorted_completions = sorted(completions, key=lambda x: x.get("rating", 0), reverse=True)
            
            # 选择最好和最差的响应
            chosen = sorted_completions[0]["response"]
            rejected = sorted_completions[-1]["response"]
            
            processed["query"].append(query)
            processed["chosen"].append(chosen)
            processed["rejected"].append(rejected)
        
        return processed
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing for GRPO"
    )
    
    # 过滤空数据
    dataset = dataset.filter(lambda x: len(x["query"]) > 0)
    
    return dataset


def setup_model_and_tokenizer(args: ScriptArguments):
    """设置模型和 tokenizer"""
    logger.info(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 量化配置
    quantization_config = None
    if args.use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


class GRPOTrainer(Trainer):
    """
    自定义 GRPO Trainer
    GRPO 核心思想：对比同一 query 的多个生成结果，优化相对排序
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        GRPO loss: 最大化 chosen 和 rejected 之间的 log probability 差异
        """
        # 获取 chosen 和 rejected 的 log probs
        chosen_inputs = {
            "input_ids": inputs["chosen_input_ids"],
            "attention_mask": inputs["chosen_attention_mask"],
            "labels": inputs["chosen_labels"],
        }
        rejected_inputs = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"],
            "labels": inputs["rejected_labels"],
        }
        
        chosen_outputs = model(**chosen_inputs)
        rejected_outputs = model(**rejected_inputs)
        
        chosen_loss = chosen_outputs.loss
        rejected_loss = rejected_outputs.loss
        
        # GRPO loss: 鼓励 chosen 的 loss 更低
        # 使用 margin-based loss
        loss = torch.nn.functional.relu(rejected_loss - chosen_loss + 0.1)
        
        return (loss, chosen_outputs) if return_outputs else loss


def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize dataset for GRPO training"""
    
    def tokenize_function(examples):
        # Tokenize chosen
        chosen_full = [q + c for q, c in zip(examples["query"], examples["chosen"])]
        chosen_encodings = tokenizer(
            chosen_full,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        
        # Tokenize rejected
        rejected_full = [q + r for q, r in zip(examples["query"], examples["rejected"])]
        rejected_encodings = tokenizer(
            rejected_full,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        
        return {
            "chosen_input_ids": chosen_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "chosen_labels": chosen_encodings["input_ids"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
            "rejected_labels": rejected_encodings["input_ids"],
        }
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def main():
    args = ScriptArguments()
    
    torch.manual_seed(args.seed)
    
    # 加载模型和数据
    model, tokenizer = setup_model_and_tokenizer(args)
    dataset = load_and_preprocess_dataset(args, tokenizer)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
    )
    
    # 创建 trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # 保存模型
    final_save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Training completed! Model saved to {final_save_path}")


if __name__ == "__main__":
    main()
