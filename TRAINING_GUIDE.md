# RLHF 训练详细指南

## 目录
1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [训练方法对比](#训练方法对比)
4. [显存优化策略](#显存优化策略)
5. [常见问题解决](#常见问题解决)
6. [进阶配置](#进阶配置)

---

## 环境准备

### 1. 硬件要求

**最低配置:**
- GPU: 4 × NVIDIA GPU (24GB+ 显存)
- CPU: 16+ 核心
- 内存: 64GB+
- 硬盘: 100GB+ 可用空间

**推荐配置:**
- GPU: 8 × NVIDIA A100 (40GB/80GB)
- CPU: 32+ 核心
- 内存: 128GB+
- 硬盘: 500GB+ SSD

### 2. 软件环境

```bash
# Python 版本
Python 3.8+

# CUDA 版本
CUDA 11.8+ 或 12.0+

# 操作系统
Linux (推荐) 或 Windows
```

### 3. 安装步骤

```bash
# 1. 克隆或下载项目
cd rlhf_training

# 2. 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 测试环境
python test_environment.py
```

---

## 快速开始

### 方式 1: 使用快速启动脚本 (Linux)

```bash
bash quick_start.sh
```

脚本会自动：
1. 检查环境
2. 安装依赖
3. 选择训练方法
4. 启动训练

### 方式 2: 手动启动

#### Linux/Mac:

```bash
# PPO 训练
bash run_multi_gpu.sh

# 或使用 Accelerate
bash run_accelerate.sh
```

#### Windows:

```cmd
REM PPO 训练
run_multi_gpu.bat
```

### 方式 3: 使用配置文件

```bash
# 1. 编辑 config.yaml 设置参数
vim config.yaml

# 2. 启动训练
python train_with_config.py --config config.yaml
```

---

## 训练方法对比

### PPO (Proximal Policy Optimization)

**优点:**
- 训练稳定，收敛性好
- 理论基础扎实
- 适合复杂任务

**缺点:**
- 需要 value model，显存占用大
- 训练速度较慢
- 实现复杂

**适用场景:**
- 有充足算力资源
- 需要高质量对齐
- 复杂的奖励函数

**启动命令:**
```bash
# 修改脚本中的 METHOD="ppo"
bash run_multi_gpu.sh
```

### GRPO (Group Relative Policy Optimization)

**优点:**
- 不需要 value model，显存占用少
- 训练速度快
- 实现简单

**缺点:**
- 理论研究较少
- 可能不如 PPO 稳定

**适用场景:**
- 显存受限
- 需要快速迭代
- 简单的对齐任务

**启动命令:**
```bash
# 修改脚本中的 METHOD="grpo"
bash run_multi_gpu.sh
```

### 方法选择建议

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 显存充足 (A100 80GB × 8) | PPO | 效果最好 |
| 显存受限 (V100 32GB × 4) | GRPO | 节省显存 |
| 快速原型验证 | GRPO | 训练快 |
| 生产环境部署 | PPO | 稳定性好 |

---

## 显存优化策略

### 优化级别 1: 基础优化 (推荐所有场景)

```yaml
# config.yaml
lora:
  enabled: true
  r: 16

quantization:
  use_8bit: true

training:
  gradient_checkpointing: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

**效果:** 显存减少约 60%

### 优化级别 2: 激进优化 (显存严重不足)

```yaml
lora:
  enabled: true
  r: 8  # 减小 rank

quantization:
  use_4bit: true  # 使用 4-bit

training:
  gradient_checkpointing: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # 增加梯度累积
  max_length: 256  # 减小序列长度
```

**效果:** 显存减少约 75%，但可能影响效果

### 优化级别 3: 极限优化 (单卡 16GB)

```yaml
lora:
  enabled: true
  r: 4
  target_modules:  # 只训练部分层
    - "q_proj"
    - "v_proj"

quantization:
  use_4bit: true

training:
  gradient_checkpointing: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  max_length: 128
```

**效果:** 可在 16GB 显存上运行，但效果会下降

### 显存占用估算表

| 配置 | Qwen3-8B 单卡显存 | 4卡总显存 |
|------|------------------|----------|
| FP16 全参数 | ~16GB | ~64GB |
| FP16 + LoRA(r=16) | ~12GB | ~48GB |
| 8bit + LoRA(r=16) | ~8GB | ~32GB |
| 4bit + LoRA(r=16) | ~6GB | ~24GB |
| 4bit + LoRA(r=8) | ~5GB | ~20GB |

---

## 常见问题解决

### 1. CUDA Out of Memory

**症状:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**

**方案 A: 减小批次大小**
```yaml
training:
  per_device_train_batch_size: 1  # 从 2 改为 1
  gradient_accumulation_steps: 16  # 从 8 改为 16
```

**方案 B: 启用更激进的量化**
```yaml
quantization:
  use_4bit: true  # 从 8bit 改为 4bit
```

**方案 C: 减小模型参数**
```yaml
lora:
  r: 8  # 从 16 改为 8
```

**方案 D: 减小序列长度**
```yaml
training:
  max_length: 256  # 从 512 改为 256
```

### 2. 训练速度慢

**症状:**
每个 batch 需要很长时间

**解决方案:**

**方案 A: 增加批次大小**
```yaml
training:
  per_device_train_batch_size: 2  # 如果显存允许
  gradient_accumulation_steps: 4  # 相应减少
```

**方案 B: 关闭 gradient checkpointing**
```yaml
training:
  gradient_checkpointing: false  # 如果显存充足
```

**方案 C: 使用更少的 PPO epochs**
```yaml
ppo:
  epochs: 2  # 从 4 改为 2
```

### 3. 分布式训练失败

**症状:**
```
RuntimeError: NCCL error
```

**解决方案:**

**检查 GPU 可见性:**
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

**设置正确的环境变量:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand
```

**检查端口占用:**
```bash
# 更换端口
torchrun --master_port=29501 ...
```

### 4. 数据集加载失败

**症状:**
```
ConnectionError: Couldn't reach https://huggingface.co
```

**解决方案:**

**方案 A: 使用镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**方案 B: 离线加载**
```bash
# 1. 先下载数据集
huggingface-cli download openbmb/UltraFeedback

# 2. 修改代码使用本地路径
dataset = load_dataset("path/to/local/dataset")
```

### 5. 模型效果不好

**可能原因和解决方案:**

**原因 1: 训练不充分**
```yaml
training:
  num_train_epochs: 3  # 增加训练轮数
```

**原因 2: 学习率不合适**
```yaml
training:
  learning_rate: 5.0e-6  # 尝试不同的学习率
```

**原因 3: LoRA rank 太小**
```yaml
lora:
  r: 32  # 增加 rank
```

**原因 4: 奖励函数太简单**
- 训练专门的 reward model
- 使用更复杂的奖励函数

---

## 进阶配置

### 1. 使用 Weights & Biases 跟踪实验

```yaml
# config.yaml
log_with: "wandb"
```

```bash
# 登录 wandb
wandb login

# 启动训练
python train_with_config.py --config config.yaml
```

### 2. 自定义 Reward Model

编辑 `train_rlhf.py`:

```python
def create_reward_model(model_name: str):
    # 加载训练好的 reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "your-reward-model"
    )
    
    def reward_fn(samples):
        # 使用 reward model 计算奖励
        inputs = tokenizer(samples, return_tensors="pt", padding=True)
        with torch.no_grad():
            rewards = reward_model(**inputs).logits.squeeze(-1)
        return rewards.tolist()
    
    return reward_fn
```

### 3. 多阶段训练

```bash
# 阶段 1: 使用小学习率预热
python train_with_config.py --config config_stage1.yaml

# 阶段 2: 使用正常学习率训练
python train_with_config.py --config config_stage2.yaml

# 阶段 3: 使用小学习率微调
python train_with_config.py --config config_stage3.yaml
```

### 4. 混合精度训练

```yaml
training:
  fp16: false
  bf16: true  # 如果 GPU 支持 BF16 (A100, H100)
```

BF16 优点:
- 数值稳定性更好
- 不容易溢出
- 适合大模型训练

### 5. 自定义数据集

```python
# 在 train_rlhf.py 中修改
def load_custom_dataset(dataset_path, tokenizer):
    # 加载自定义数据
    dataset = load_dataset("json", data_files=dataset_path)
    
    def preprocess(examples):
        # 自定义预处理逻辑
        queries = [format_query(q) for q in examples["query"]]
        responses = examples["response"]
        return {"query": queries, "response": responses}
    
    return dataset.map(preprocess, batched=True)
```

---

## 训练监控

### 1. TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir=./output/logs --port=6006

# 浏览器访问
http://localhost:6006
```

### 2. 实时监控 GPU

```bash
# 实时查看 GPU 使用情况
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

### 3. 训练日志

```bash
# 查看训练日志
tail -f output/logs/training.log

# 搜索错误
grep -i error output/logs/training.log
```

---

## 最佳实践

### 1. 训练前

- [ ] 运行 `python test_environment.py` 检查环境
- [ ] 在小数据集上测试完整流程
- [ ] 确认有足够的磁盘空间
- [ ] 设置好日志和监控

### 2. 训练中

- [ ] 定期检查 GPU 利用率
- [ ] 监控训练损失曲线
- [ ] 保存多个 checkpoint
- [ ] 定期在验证集上评估

### 3. 训练后

- [ ] 评估模型效果
- [ ] 对比不同 checkpoint
- [ ] 进行人工评估
- [ ] 记录实验结果

---

## 参考资源

- [TRL 文档](https://huggingface.co/docs/trl)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Qwen 模型](https://github.com/QwenLM/Qwen)
- [UltraFeedback 数据集](https://huggingface.co/datasets/openbmb/UltraFeedback)
- [RLHF 论文](https://arxiv.org/abs/2203.02155)

---

## 获取帮助

如果遇到问题:

1. 查看本指南的常见问题部分
2. 检查 GitHub Issues
3. 查看相关文档
4. 在社区提问

祝训练顺利！🚀
