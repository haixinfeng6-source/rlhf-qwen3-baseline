# RLHF Baseline 训练项目

基于 Qwen3-8B 和 UltraFeedback 数据集的 RLHF (Reinforcement Learning from Human Feedback) 训练代码。

## 项目结构

```
.
├── train_rlhf.py              # PPO 训练脚本
├── train_grpo.py              # GRPO 训练脚本
├── evaluate_model.py          # 模型评估脚本
├── run_multi_gpu.sh           # 多卡训练启动脚本 (torchrun)
├── run_accelerate.sh          # Accelerate 启动脚本
├── accelerate_config.yaml     # Accelerate 配置文件
├── requirements.txt           # Python 依赖
└── README.md                  # 本文件
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证 GPU 环境

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 训练方法

### 方法 1: PPO (Proximal Policy Optimization)

PPO 是经典的 RLHF 方法，使用 actor-critic 架构。

**特点:**
- 需要 value model (额外的显存开销)
- 训练稳定性好
- 适合复杂任务

**启动训练:**

```bash
# 使用 torchrun (推荐)
bash run_multi_gpu.sh

# 或使用 accelerate
bash run_accelerate.sh
```

### 方法 2: GRPO (Group Relative Policy Optimization)

GRPO 是更简单高效的方法，不需要 value model。

**特点:**
- 显存占用更少
- 训练速度更快
- 适合资源受限场景

**启动训练:**

修改脚本中的 `METHOD="grpo"` 然后运行:

```bash
bash run_multi_gpu.sh
```

## 显存优化技巧

### 1. 量化 (Quantization)

**8-bit 量化** (推荐):
```python
--use_8bit
```
- 显存减少约 50%
- 精度损失很小
- 训练速度略慢

**4-bit 量化** (极限优化):
```python
--use_4bit
```
- 显存减少约 75%
- 可能影响精度
- 适合显存严重不足时

### 2. LoRA (Low-Rank Adaptation)

只训练少量参数，大幅减少显存:

```python
--use_lora \
--lora_r=16 \          # rank 越小显存越少，但表达能力下降
--lora_alpha=32 \
--lora_dropout=0.05
```

**LoRA 参数建议:**
- `lora_r`: 8-64，推荐 16
- `lora_alpha`: 通常设为 `lora_r * 2`
- `target_modules`: 选择要训练的层

### 3. Gradient Checkpointing

用计算换显存:

```python
--gradient_checkpointing
```

- 显存减少 30-50%
- 训练速度降低 20-30%
- 几乎无精度损失

### 4. 批次大小优化

```python
--per_device_train_batch_size=1 \      # 单卡批次大小
--gradient_accumulation_steps=8        # 梯度累积步数
```

**有效批次大小** = `batch_size × gradient_accumulation_steps × num_gpus`

例如: 1 × 8 × 4 = 32

### 5. 序列长度控制

```python
--max_length=512    # 减少序列长度可显著降低显存
```

### 显存占用估算

以 Qwen3-8B 为例 (4卡训练):

| 配置 | 单卡显存 | 说明 |
|------|---------|------|
| 全精度 + 全参数 | ~32GB | 不可行 |
| FP16 + 全参数 | ~16GB | 勉强可行 |
| 8bit + LoRA | ~8GB | 推荐 |
| 4bit + LoRA | ~6GB | 极限优化 |

## 训练参数说明

### 核心参数

```bash
--model_name="Qwen/Qwen2.5-7B-Instruct"    # 模型名称
--dataset_name="openbmb/UltraFeedback"     # 数据集
--output_dir="./output"                     # 输出目录
--num_train_epochs=1                        # 训练轮数
--learning_rate=1e-5                        # 学习率
```

### PPO 特定参数

```bash
--ppo_epochs=4              # PPO 内部迭代次数
--mini_batch_size=1         # PPO mini-batch 大小
```

### GRPO 特定参数

```bash
--num_generations=4         # 每个 query 生成的响应数
--temperature=0.8           # 生成温度
```

## 多卡训练配置

### 4 卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
```

### 8 卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
```

修改 `accelerate_config.yaml` 中的 `num_processes`。

## 模型评估

训练完成后评估模型:

```bash
python evaluate_model.py \
    --model_path=./output/final_model \
    --base_model=Qwen/Qwen2.5-7B-Instruct \  # 如果使用 LoRA
    --output_file=results.json
```

## 监控训练

### TensorBoard

```bash
tensorboard --logdir=./output/logs
```

### Weights & Biases (可选)

在代码中设置:
```python
log_with="wandb"
```

## 常见问题

### 1. CUDA Out of Memory

**解决方案:**
- 减小 `batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing`
- 使用更激进的量化 (4bit)
- 减小 `max_length`
- 减小 `lora_r`

### 2. 训练速度慢

**优化方案:**
- 增加 `batch_size` (如果显存允许)
- 减少 `gradient_accumulation_steps`
- 关闭 `gradient_checkpointing`
- 使用更少的 `ppo_epochs`

### 3. 模型效果不好

**改进方向:**
- 增加训练数据量
- 调整学习率
- 增加训练轮数
- 使用更大的 LoRA rank
- 改进 reward model

### 4. 分布式训练失败

**检查项:**
- 确认所有 GPU 可见: `nvidia-smi`
- 检查端口是否被占用
- 确认 NCCL 环境变量设置正确
- 查看错误日志

## 性能优化建议

### 训练速度优化

1. **使用 Flash Attention** (如果支持):
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
)
```

2. **混合精度训练**:
```python
fp16=True  # 或 bf16=True (如果 GPU 支持)
```

3. **数据加载优化**:
```python
dataloader_num_workers=4
dataloader_pin_memory=True
```

### 显存优化优先级

1. 启用 8-bit 量化
2. 使用 LoRA
3. 启用 gradient checkpointing
4. 减小 batch size，增加梯度累积
5. 减小序列长度
6. 考虑 4-bit 量化

## 下一步

1. **改进 Reward Model**: 当前使用简单的长度奖励，应训练专门的 reward model
2. **数据增强**: 使用更多高质量数据
3. **超参数调优**: 使用 wandb sweep 等工具
4. **模型融合**: 尝试多个 checkpoint 的融合
5. **在线评估**: 部署模型进行人工评估

## 参考资料

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Qwen Documentation](https://github.com/QwenLM/Qwen)
- [UltraFeedback Dataset](https://huggingface.co/datasets/openbmb/UltraFeedback)

## License

MIT
