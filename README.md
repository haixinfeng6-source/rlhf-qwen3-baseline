# RLHF Baseline 训练项目

基于 Qwen 和 UltraFeedback 数据集的 RLHF (Reinforcement Learning from Human Feedback) 训练代码。

使用标准 **PPO (Proximal Policy Optimization)** 方法，基于 HuggingFace TRL 官方实现。

## 📁 项目结构

```
.
├── train_rlhf.py              # RLHF (PPO) 训练脚本
├── evaluate_model.py          # 模型评估脚本
├── test_environment.py        # 环境测试脚本
├── run_multi_gpu.sh           # 多卡训练启动脚本 (Linux)
├── run_multi_gpu.bat          # 多卡训练启动脚本 (Windows)
├── run_accelerate.sh          # Accelerate 启动脚本
├── config.yaml                # 训练配置文件
├── config_local_test.yaml     # 测试配置文件
├── requirements.txt           # Python 依赖
├── deploy.sh                  # 自动部署脚本 (Linux)
├── deploy.bat                 # 自动部署脚本 (Windows)
└── README.md                  # 本文件
```

## 🎯 RLHF 方法说明

本项目使用 **PPO (Proximal Policy Optimization)** 进行 RLHF 训练。

### 为什么选择 PPO？

- ✅ **业界标准**: ChatGPT, GPT-4, Claude 等都使用此方法
- ✅ **效果最佳**: 训练稳定，对齐效果好
- ✅ **理论成熟**: 有大量研究和实践支持
- ✅ **官方支持**: HuggingFace TRL 官方实现

### 技术架构

```
RLHF Pipeline:
1. Policy Model (Actor) - 生成响应
2. Value Model (Critic) - 评估状态价值
3. Reward Model - 计算奖励信号
4. PPO Algorithm - 优化策略
```

### 参考资料

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证 GPU
python test_environment.py
```

### 2. 启动训练

**Linux:**
```bash
bash run_multi_gpu.sh
```

**Windows:**
```cmd
run_multi_gpu.bat
```

**或直接运行:**
```bash
python train_rlhf.py
```

### 3. 自定义配置

编辑 `config.yaml` 或使用命令行参数：

```bash
python train_rlhf.py \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --dataset_name=openbmb/UltraFeedback \
    --output_dir=./output \
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing
```

## 💾 显存优化

### 优化策略

本项目内置多种显存优化技术：

1. **8-bit/4-bit 量化** - 减少 50-75% 显存
2. **LoRA** - 只训练少量参数
3. **Gradient Checkpointing** - 用计算换显存
4. **梯度累积** - 小批次模拟大批次

### 显存占用估算 (Qwen-8B)

| 配置 | 单卡显存 | 4卡总显存 |
|------|---------|----------|
| FP16 全参数 | ~16GB | ~64GB |
| 8bit + LoRA(r=16) | ~8GB | ~32GB |
| 4bit + LoRA(r=16) | ~6GB | ~24GB |
| 4bit + LoRA(r=8) | ~5GB | ~20GB |

### 显存不足？

编辑 `config.yaml`:

```yaml
quantization:
  use_4bit: true  # 使用 4-bit 量化

lora:
  r: 8  # 减小 LoRA rank

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # 增加梯度累积
  max_length: 256  # 减小序列长度
```

## 📊 训练参数说明

### 核心参数

```bash
--model_name              # 模型名称或路径
--dataset_name            # 数据集名称
--output_dir              # 输出目录
--num_train_epochs        # 训练轮数
--learning_rate           # 学习率
```

### PPO 参数

```bash
--ppo_epochs              # PPO 内部迭代次数 (默认 4)
--mini_batch_size         # PPO mini-batch 大小 (默认 1)
```

### 优化参数

```bash
--use_lora                # 启用 LoRA
--lora_r                  # LoRA rank (默认 16)
--use_8bit                # 8-bit 量化
--use_4bit                # 4-bit 量化
--gradient_checkpointing  # Gradient checkpointing
```

## 🖥️ 多卡训练

### 4 卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash run_multi_gpu.sh
```

### 8 卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 修改 run_multi_gpu.sh 中的 NUM_GPUS=8
bash run_multi_gpu.sh
```

## 📈 监控训练

### TensorBoard

```bash
tensorboard --logdir=./output/logs --port=6006
```

### 实时 GPU 监控

```bash
watch -n 1 nvidia-smi
```

## 🎓 模型评估

训练完成后评估模型：

```bash
python evaluate_model.py \
    --model_path=./output/final_model \
    --base_model=Qwen/Qwen2.5-7B-Instruct \
    --output_file=results.json
```

## 🌐 部署到开发机

### 在线部署

如果开发机可以联网：

**自动部署 (推荐):**

**Linux:**
```bash
# 1. 编辑配置
vim deploy.sh
# 修改: SERVER_IP, USERNAME

# 2. 运行部署
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```cmd
REM 1. 编辑配置
notepad deploy.bat
REM 修改: SERVER_IP, USERNAME

REM 2. 运行部署
deploy.bat
```

**手动部署:**

```bash
# 1. 上传代码
scp -r . username@server:/path/to/project

# 2. SSH 登录
ssh username@server

# 3. 安装环境
cd /path/to/project
pip install -r requirements.txt

# 4. 启动训练
bash run_multi_gpu.sh
```

### 离线部署（开发机无法联网）

如果开发机无法联网，需要提前下载资源。详细步骤请查看 `OFFLINE_SETUP.md`

**快速流程:**

1. **在有网络的机器上下载资源:**
```bash
# 下载模型、数据集和依赖包
python download_assets.py

# 下载 Python 依赖包
pip download -r requirements.txt -d offline_assets/wheels
```

2. **传输到开发机:**
```bash
# 压缩
tar -czf offline_assets.tar.gz offline_assets/

# 传输
scp offline_assets.tar.gz username@server:/path/to/project/
```

3. **在开发机上安装:**
```bash
# 解压
tar -xzf offline_assets.tar.gz

# 安装依赖
bash install_offline.sh

# 测试
python test_offline_assets.py

# 启动训练
python train_rlhf.py \
    --model_name=./offline_assets/models/Qwen--Qwen2.5-7B-Instruct \
    --dataset_name=./offline_assets/datasets/openbmb--UltraFeedback \
    --output_dir=./output_offline
```

详细步骤请查看 `OFFLINE_SETUP.md`

## 🐛 常见问题

### 1. CUDA Out of Memory

**解决方案:**
- 使用 4-bit 量化: `--use_4bit`
- 减小批次大小: `--per_device_train_batch_size=1`
- 增加梯度累积: `--gradient_accumulation_steps=16`
- 减小序列长度: `--max_length=256`

### 2. 数据集下载失败

**解决方案:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 多卡训练失败

**解决方案:**
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

更多问题请查看 `TRAINING_GUIDE.md`

## 📚 文档

- `README.md` - 项目概述 (本文件)
- `TRAINING_GUIDE.md` - 详细训练指南
- `LOCAL_SETUP.md` - 本地运行指南
- `DEPLOY_TO_SERVER.md` - 开发机部署指南
- `OFFLINE_SETUP.md` - 离线环境部署指南 ⭐
- `QUICK_REFERENCE.md` - 快速参考卡片

## 🔗 相关链接

- GitHub: https://github.com/haixinfeng6-source/rlhf-qwen3-baseline
- HuggingFace TRL: https://github.com/huggingface/trl
- Qwen: https://github.com/QwenLM/Qwen
- UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback

## 📄 License

MIT

## 🙏 致谢

本项目基于以下开源项目：

- [HuggingFace TRL](https://github.com/huggingface/trl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [Qwen](https://github.com/QwenLM/Qwen)
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
