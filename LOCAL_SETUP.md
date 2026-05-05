# 本地运行指南

## 📋 目录
1. [环境要求](#环境要求)
2. [安装步骤](#安装步骤)
3. [快速测试](#快速测试)
4. [本地训练](#本地训练)
5. [常见问题](#常见问题)

---

## 环境要求

### 最低配置（可以运行测试）
- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **内存**: 16GB+
- **硬盘**: 50GB+ 可用空间
- **系统**: Windows 10/11, Linux, macOS (with NVIDIA GPU)

### 推荐配置（完整训练）
- **GPU**: NVIDIA RTX 3090/4090 (24GB) 或 A100 (40GB/80GB)
- **内存**: 32GB+
- **硬盘**: 100GB+ SSD
- **系统**: Linux (Ubuntu 20.04+)

### 软件要求
- Python 3.8+
- CUDA 11.8+ 或 12.0+
- Git

---

## 安装步骤

### 1. 克隆仓库

```bash
# 克隆项目
git clone https://github.com/haixinfeng6-source/rlhf-qwen3-baseline.git

# 进入目录
cd rlhf-qwen3-baseline
```

### 2. 创建虚拟环境（推荐）

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 如果下载慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 验证安装

```bash
# 运行环境测试
python test_environment.py
```

---

## 快速测试

### 测试 1: 环境检查

```bash
python run_local_test.py
```

这个脚本会：
- ✅ 检查 GPU 是否可用
- ✅ 测试模型加载
- ✅ 测试数据集加载
- ✅ 估算显存需求

### 测试 2: 小规模训练测试

使用测试配置文件，只训练 50 条数据：

```bash
python train_with_config.py --config config_local_test.yaml
```

预计运行时间：5-10 分钟（取决于 GPU）

---

## 本地训练

### 方式 1: 单卡训练（推荐本地使用）

```bash
# 使用默认配置
python train_rlhf.py

# 或使用配置文件
python train_with_config.py --config config.yaml
```

### 方式 2: 多卡训练（如果有多张 GPU）

**Linux:**
```bash
# 修改 run_multi_gpu.sh 中的 NUM_GPUS
bash run_multi_gpu.sh
```

**Windows:**
```cmd
REM 修改 run_multi_gpu.bat 中的 NUM_GPUS
run_multi_gpu.bat
```

### 方式 3: 自定义配置

编辑 `config.yaml` 调整参数：

```yaml
# 显存不足时的配置
quantization:
  use_4bit: true  # 使用 4-bit 量化

lora:
  r: 8  # 减小 LoRA rank

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  max_length: 256  # 减小序列长度
```

---

## 显存优化建议

### 如果你的 GPU 显存是...

#### 24GB+ (RTX 3090, 4090, A100 40GB)
```yaml
quantization:
  use_8bit: true

lora:
  r: 16

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_length: 512
```

#### 16GB (RTX 4080, V100 16GB)
```yaml
quantization:
  use_4bit: true

lora:
  r: 8

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  max_length: 256
```

#### 12GB (RTX 3080, 3060)
```yaml
quantization:
  use_4bit: true

lora:
  r: 4
  target_modules:  # 只训练部分层
    - "q_proj"
    - "v_proj"

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  max_length: 128
```

#### < 12GB
❌ 不建议在本地训练 Qwen3-8B
建议使用云服务器或更小的模型

---

## 常见问题

### 1. 没有 GPU 可以运行吗？

**可以运行测试代码**，但不建议实际训练：
- CPU 训练会非常慢（可能需要几天甚至几周）
- 可以用于验证代码逻辑
- 建议使用云服务器（如 AutoDL, 恒源云）

### 2. 如何使用国内镜像加速？

**HuggingFace 镜像:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**PyPI 镜像:**
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. CUDA Out of Memory 怎么办？

**方案 A: 减小批次大小**
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # 增加这个值
```

**方案 B: 使用更激进的量化**
```yaml
quantization:
  use_4bit: true  # 从 8bit 改为 4bit
```

**方案 C: 减小序列长度**
```yaml
training:
  max_length: 256  # 从 512 改为 256
```

**方案 D: 减小 LoRA rank**
```yaml
lora:
  r: 8  # 从 16 改为 8
```

### 4. 数据集下载失败

**使用镜像:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python train_rlhf.py
```

**离线下载:**
```bash
# 先下载数据集
huggingface-cli download openbmb/UltraFeedback

# 修改代码使用本地路径
```

### 5. 训练速度太慢

**检查项:**
- ✅ 确认使用了 GPU（不是 CPU）
- ✅ 关闭 gradient_checkpointing（如果显存充足）
- ✅ 增加 batch_size（如果显存允许）
- ✅ 使用 FP16 或 BF16 混合精度

### 6. 如何监控训练进度？

**使用 TensorBoard:**
```bash
# 启动 TensorBoard
tensorboard --logdir=./output/logs

# 浏览器访问
http://localhost:6006
```

**实时查看 GPU:**
```bash
# Linux
watch -n 1 nvidia-smi

# Windows (PowerShell)
while($true) { nvidia-smi; sleep 1; cls }
```

---

## 云服务器推荐

如果本地显存不足，推荐使用云服务器：

### 国内平台
1. **AutoDL** (https://www.autodl.com)
   - RTX 4090 (24GB): ~2.5元/小时
   - A100 (40GB): ~4元/小时

2. **恒源云** (https://gpushare.com)
   - RTX 3090 (24GB): ~2元/小时
   - A100 (80GB): ~6元/小时

3. **阿里云 PAI**
   - 按需付费
   - 适合长期使用

### 国际平台
1. **Google Colab Pro**
   - A100 (40GB)
   - $10/月

2. **AWS SageMaker**
   - 按需付费
   - 适合企业用户

---

## 训练时间估算

以 Qwen3-8B + UltraFeedback 全量数据为例：

| GPU | 配置 | 预计时间 |
|-----|------|---------|
| RTX 4090 (24GB) | 8bit + LoRA | ~12-16 小时 |
| A100 (40GB) | 8bit + LoRA | ~8-10 小时 |
| A100 (80GB) × 4 | 8bit + LoRA | ~2-3 小时 |

*实际时间取决于具体配置和数据量*

---

## 下一步

1. ✅ 运行 `python test_environment.py` 检查环境
2. ✅ 运行 `python run_local_test.py` 快速测试
3. ✅ 使用 `config_local_test.yaml` 进行小规模训练
4. ✅ 调整配置文件适配你的硬件
5. ✅ 开始完整训练

祝训练顺利！🚀

---

## 获取帮助

- 查看 `TRAINING_GUIDE.md` 了解详细训练指南
- 查看 `README.md` 了解项目概述
- GitHub Issues: https://github.com/haixinfeng6-source/rlhf-qwen3-baseline/issues
