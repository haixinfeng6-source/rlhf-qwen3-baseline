# RLHF Baseline for Qwen3-8B

基于 Qwen3-8B 和 UltraFeedback 数据集的 RLHF (Reinforcement Learning from Human Feedback) 训练项目。

## 🚀 快速开始

### 1. 环境准备

**Python 版本要求: 3.8-3.11** (推荐 3.10)
> 注意: Python 3.12+ 可能有不兼容问题，建议使用 3.10 或 3.11

```bash
# 安装依赖
pip install -r requirements.txt

# 如果网络问题，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 启动训练
```bash
# 单卡训练
python train_rlhf.py

# 多卡训练 (4卡)
bash run_multi_gpu.sh
```

### 3. 自定义配置
编辑 `config.yaml` 或使用命令行参数：
```bash
python train_rlhf.py \
    --model_name=Qwen/Qwen3-8B \
    --dataset_name=openbmb/UltraFeedback \
    --output_dir=./output \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing
```

## 📦 离线运行（无网络集群）

### 1. 下载资源
```bash
python download_assets.py
pip download -r requirements.txt -d offline_assets/wheels
```

### 2. 传输到集群
```bash
rsync -avz offline_assets/ username@server:/path/to/project/
```

### 3. 在集群上运行
```bash
bash install_offline.sh
bash run_offline.sh
```

## 🔧 显存优化

| 配置 | 单卡显存 | 说明 |
|------|---------|------|
| FP16 全参数 | ~16GB | 基础配置 |
| 8bit + LoRA | ~8GB | 推荐配置 |
| 4bit + LoRA | ~6GB | 低显存配置 |

**优化参数：**
- `--use_8bit`: 8-bit 量化
- `--use_lora`: LoRA 微调
- `--gradient_checkpointing`: 梯度检查点
- `--per_device_train_batch_size=1`: 小批次
- `--gradient_accumulation_steps=8`: 梯度累积

## 📚 文档

- `TRAINING_GUIDE.md` - 详细训练指南
- `OFFLINE_SETUP.md` - 离线部署指南
- `QUICK_REFERENCE.md` - 快速参考

## 🐛 常见问题

### 1. Python 版本问题
```bash
# 如果使用 Python 3.12+ 遇到导入错误
# 建议降级到 Python 3.10 或 3.11
conda create -n rlhf python=3.10
conda activate rlhf
```

### 2. CUDA Out of Memory
```bash
# 使用 4-bit 量化
python train_rlhf.py --use_4bit

# 减小批次大小
python train_rlhf.py --per_device_train_batch_size=1 --gradient_accumulation_steps=16
```

### 3. 数据集下载失败
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 4. 多卡训练失败
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# 如果所有进程都立即失败，尝试单进程测试
python train_rlhf.py --max_samples=10  # 测试小样本
```

### 5. TRL 导入错误
```bash
# 确保 TRL 版本 >= 0.11.0
pip install trl>=0.11.0 rich>=14.0.0

# 如果仍有导入错误，检查导入语句
# 正确: from trl.models import AutoModelForCausalLMWithValueHead
# 错误: from trl import AutoModelForCausalLMWithValueHead
```

## 📄 License

MIT

## 🙏 致谢

- [HuggingFace TRL](https://github.com/huggingface/trl)
- [Qwen](https://github.com/QwenLM/Qwen)
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
