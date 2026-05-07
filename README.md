# RLHF Baseline for Qwen3-8B

基于 Qwen3-8B 和 UltraFeedback 数据集的 RLHF (Reinforcement Learning from Human Feedback) 训练项目。

## 🚀 快速开始

### 1. 环境准备
```bash
pip install -r requirements.txt
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

### 1. CUDA Out of Memory
```bash
# 使用 4-bit 量化
python train_rlhf.py --use_4bit

# 减小批次大小
python train_rlhf.py --per_device_train_batch_size=1 --gradient_accumulation_steps=16
```

### 2. 数据集下载失败
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 多卡训练失败
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

## 📄 License

MIT

## 🙏 致谢

- [HuggingFace TRL](https://github.com/huggingface/trl)
- [Qwen](https://github.com/QwenLM/Qwen)
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
