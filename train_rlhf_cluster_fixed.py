"""
RLHF 训练脚本 - 集群修复版本
专为多GPU集群环境设计，解决分布式训练问题

主要修复:
1. 简化 TRL 导入逻辑
2. 修复分布式训练初始化
3. 支持离线资源路径
4. 添加详细的错误处理
"""

import os
import sys
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 检查环境
print("=" * 60)
print("RLHF 集群训练环境检查")
print("=" * 60)

# 检查 PyTorch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")

# 检查分布式环境
local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", 1))
print(f"分布式环境: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

# 导入检查
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
    print("✓ Transformers 导入成功")
except ImportError as e:
    print(f"✗ Transformers 导入失败: {e}")
    sys.exit(1)

# TRL 导入 - 简化版本
try:
    # 尝试标准导入
    from trl import PPOConfig, PPOTrainer
    print("✓ TRL PPOConfig, PPOTrainer 导入成功")
except ImportError as e:
    print(f"✗ TRL 标准导入失败: {e}")
    sys.exit(1)

# 检查 TRL 版本并导入 AutoModelForCausalLMWithValueHead
try:
    import importlib.metadata
    trl_version = importlib.metadata.version("trl")
    print(f"TRL 版本: {trl_version}")
    
    if trl_version.startswith('1.'):
        # TRL 1.x 版本
        try:
            from trl import AutoModelForCausalLMWithValueHead
            print("✓ TRL 1.x AutoModelForCausalLMWithValueHead 导入成功")
        except ImportError:
            try:
                from trl.models import AutoModelForCausalLMWithValueHead
                print("✓ TRL 1.x models.AutoModelForCausalLMWithValueHead 导入成功")
            except ImportError as e:
                print(f"✗ TRL 1.x 导入失败: {e}")
                sys.exit(1)
    else:
        # TRL 0.x 版本
        try:
            from trl.models import AutoModelForCausalLMWithValueHead
            print("✓ TRL 0.x AutoModelForCausalLMWithValueHead 导入成功")
        except ImportError:
            try:
                from trl import AutoModelForCausalLMWithValueHead
                print("✓ TRL 0.x 直接导入成功")
            except ImportError as e:
                print(f"✗ TRL 0.x 导入失败: {e}")
                sys.exit(1)
except importlib.metadata.PackageNotFoundError:
    print("✗ TRL 未安装")
    sys.exit(1)

# 其他导入
try:
    from datasets import load_dataset, load_from_disk
    print("✓ Datasets 导入成功")
except ImportError as e:
    print(f"✗ Datasets 导入失败: {e}")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    print("✓ PEFT 导入成功")
except ImportError as e:
    print(f"✗ PEFT 导入失败: {e}")
    sys.exit(1)

print("=" * 60)
print("环境检查完成")
print("=" * 60)


@dataclass
class TrainingArguments:
    """RLHF 训练参数配置"""
    
    # 模型配置
    model_name: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "预训练模型名称或路径"}
    )
    
    # 数据集配置
    dataset_name: str = field(
        default="openbmb/UltraFeedback",
        metadata={"help": "数据集名称或路径"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "数据集分割"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "最大样本数"}
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
        default="./output_rlhf_cluster",
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
    """加载并预处理数据集"""
    logger.info(f"加载数据集: {args.dataset_name}")
    
    # 判断是本地路径还是 HuggingFace 数据集
    if os.path.exists(args.dataset_name):
        # 从本地加载
        logger.info("从本地路径加载数据集")
        dataset = load_from_disk(args.dataset_name)
        if hasattr(dataset, 'keys'):
            if args.dataset_split in dataset:
                dataset = dataset[args.dataset_split]
            else:
                dataset = dataset[list(dataset.keys())[0]]
    else:
        # 从 HuggingFace Hub 加载
        logger.info("从 HuggingFace Hub 加载数据集")
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 简化预处理 - 只提取 instruction
    def preprocess_function(examples):
        queries = []
        for instruction in examples["instruction"]:
            messages = [{"role": "user", "content": instruction}]
            query = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            queries.append(query)
        
        return {"query": queries}
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="预处理数据集"
    )
    
    return dataset


def create_reward_function():
    """创建简化的奖励函数"""
    def reward_fn(samples: list) -> list:
        rewards = []
        for sample in samples:
            # 简单奖励：基于长度
            length = len(sample.split())
            reward = min(length / 50, 1.0)
            rewards.append(reward)
        return rewards
    return reward_fn


def setup_model_and_tokenizer(args: TrainingArguments):
    """设置模型和 tokenizer"""
    logger.info(f"加载模型: {args.model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing 已启用")
    
    # LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA 已启用")
    
    # 包装为 PPO 模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    return model, tokenizer


def main():
    """主训练函数"""
    # 分布式初始化
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank != -1:
        print(f"[进程 {local_rank}] 初始化分布式训练...")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    
    # 解析参数
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建输出目录（只在主进程）
    if local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和数据集
    try:
        model, tokenizer = setup_model_and_tokenizer(args)
        dataset = load_and_preprocess_dataset(args, tokenizer)
    except Exception as e:
        logger.error(f"加载失败: {e}")
        if local_rank != -1:
            torch.distributed.destroy_process_group()
        raise
    
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
    try:
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
        )
    except Exception as e:
        logger.error(f"创建 PPO Trainer 失败: {e}")
        if local_rank != -1:
            torch.distributed.destroy_process_group()
        raise
    
    # 奖励函数
    reward_fn = create_reward_function()
    
    logger.info("=" * 60)
    logger.info("开始 RLHF 训练")
    logger.info(f"模型: {args.model_name}")
    logger.info(f"数据集: {args.dataset_name}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)
    
    # 简化训练循环
    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            if batch_idx >= 10:  # 只训练前10个batch用于测试
                break
                
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
            
            if batch_idx % args.logging_steps == 0:
                logger.info(f"Batch {batch_idx}: {stats}")
    
    # 保存模型
    if local_rank in [-1, 0]:
        final_save_path = os.path.join(args.output_dir, "final_model")
        ppo_trainer.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"模型已保存到 {final_save_path}")
    
    # 清理
    if local_rank != -1:
        torch.distributed.destroy_process_group()
    
    logger.info("训练完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理分布式环境
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank != -1:
            try:
                torch.distributed.destroy_process_group()
            except:
                pass
        
        sys.exit(1)