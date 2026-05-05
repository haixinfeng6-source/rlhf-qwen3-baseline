# 离线环境部署指南

开发机无法联网时，需要提前下载模型、数据集和依赖包。

## 📋 目录
1. [准备工作](#准备工作)
2. [下载资源](#下载资源)
3. [传输到开发机](#传输到开发机)
4. [离线安装](#离线安装)
5. [离线训练](#离线训练)

---

## 1. 准备工作

### 在有网络的机器上操作

确保已安装必要的工具：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 安装 datasets
pip install datasets

# 克隆项目
git clone https://github.com/haixinfeng6-source/rlhf-qwen3-baseline.git
cd rlhf-qwen3-baseline
```

---

## 2. 下载资源

### 方式 1: 使用自动下载脚本（推荐）

```bash
# 下载所有资源（模型 + 数据集）
python download_assets.py

# 或指定模型和数据集
python download_assets.py \
    --model=Qwen/Qwen2.5-7B-Instruct \
    --dataset=openbmb/UltraFeedback

# 只下载模型
python download_assets.py --skip-dataset

# 只下载数据集
python download_assets.py --skip-model
```

### 方式 2: 手动下载

#### 2.1 下载模型

```python
# download_model.py
from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen2.5-7B-Instruct"
save_path = "./offline_assets/models/Qwen--Qwen2.5-7B-Instruct"

snapshot_download(
    repo_id=model_name,
    local_dir=save_path,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"模型已下载到: {save_path}")
```

运行：
```bash
python download_model.py
```

#### 2.2 下载数据集

```python
# download_dataset.py
from datasets import load_dataset

dataset_name = "openbmb/UltraFeedback"
save_path = "./offline_assets/datasets/openbmb--UltraFeedback"

# 下载数据集
dataset = load_dataset(dataset_name)

# 保存到本地
dataset.save_to_disk(save_path)

print(f"数据集已下载到: {save_path}")
print(f"训练集样本数: {len(dataset['train'])}")
```

运行：
```bash
python download_dataset.py
```

#### 2.3 下载 Python 依赖包

```bash
# 创建目录
mkdir -p offline_assets/wheels

# 下载所有依赖包
pip download -r requirements.txt -d offline_assets/wheels

# 或使用国内镜像加速
pip download -r requirements.txt -d offline_assets/wheels \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 下载完成后的目录结构

```
offline_assets/
├── models/
│   └── Qwen--Qwen2.5-7B-Instruct/
│       ├── config.json
│       ├── model-00001-of-00004.safetensors
│       ├── model-00002-of-00004.safetensors
│       ├── model-00003-of-00004.safetensors
│       ├── model-00004-of-00004.safetensors
│       ├── tokenizer.json
│       └── ...
├── datasets/
│   └── openbmb--UltraFeedback/
│       ├── train/
│       │   ├── data-00000-of-00001.arrow
│       │   └── state.json
│       └── dataset_info.json
└── wheels/
    ├── torch-2.0.0-cp310-cp310-linux_x86_64.whl
    ├── transformers-4.36.0-py3-none-any.whl
    ├── trl-0.7.0-py3-none-any.whl
    └── ...
```

---

## 3. 传输到开发机

### 方式 1: 使用 SCP

```bash
# 压缩资源（可选，加快传输）
tar -czf offline_assets.tar.gz offline_assets/

# 传输到开发机
scp offline_assets.tar.gz username@server:/path/to/project/

# 在开发机上解压
ssh username@server
cd /path/to/project
tar -xzf offline_assets.tar.gz
```

### 方式 2: 使用 rsync（推荐，支持断点续传）

```bash
# 同步整个目录
rsync -avz --progress offline_assets/ \
    username@server:/path/to/project/offline_assets/

# 如果使用 SSH 密钥
rsync -avz --progress -e "ssh -i /path/to/key.pem" \
    offline_assets/ \
    username@server:/path/to/project/offline_assets/
```

### 方式 3: 使用移动硬盘

如果网络传输太慢，可以：

1. 将 `offline_assets` 目录复制到移动硬盘
2. 物理传输到开发机
3. 从移动硬盘复制到项目目录

---

## 4. 离线安装

### 在开发机上操作

#### 4.1 安装 Python 依赖

```bash
# 进入项目目录
cd /path/to/project

# 离线安装依赖
pip install --no-index --find-links=offline_assets/wheels -r requirements.txt

# 或使用自动安装脚本
bash install_offline.sh
```

#### 4.2 验证安装

```bash
# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证 Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 验证 TRL
python -c "import trl; print(f'TRL: {trl.__version__}')"

# 验证 PEFT
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

#### 4.3 验证模型和数据集

```python
# test_offline_assets.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

# 测试模型加载
model_path = "./offline_assets/models/Qwen--Qwen2.5-7B-Instruct"
print(f"加载模型: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"✓ Tokenizer 加载成功")

# 测试数据集加载
dataset_path = "./offline_assets/datasets/openbmb--UltraFeedback"
print(f"\n加载数据集: {dataset_path}")

dataset = load_from_disk(dataset_path)
print(f"✓ 数据集加载成功")
print(f"  训练集样本数: {len(dataset['train'])}")

print("\n所有资源验证通过！")
```

运行验证：
```bash
python test_offline_assets.py
```

---

## 5. 离线训练

### 5.1 修改训练脚本使用本地路径

创建离线训练配置文件 `config_offline.yaml`:

```yaml
# 离线训练配置

model:
  name: "./offline_assets/models/Qwen--Qwen2.5-7B-Instruct"
  trust_remote_code: true

dataset:
  name: "./offline_assets/datasets/openbmb--UltraFeedback"
  split: "train"
  max_samples: null

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

quantization:
  use_8bit: true
  use_4bit: false

training:
  output_dir: "./output_offline"
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  max_grad_norm: 1.0
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  save_total_limit: 3
  max_length: 512
  gradient_checkpointing: true
  fp16: true

ppo:
  epochs: 4
  mini_batch_size: 1

distributed:
  num_gpus: 4

seed: 42
log_with: "tensorboard"
```

### 5.2 修改训练脚本加载本地数据

修改 `train_rlhf.py` 中的数据加载函数：

```python
def load_and_preprocess_dataset(args: TrainingArguments, tokenizer):
    """加载并预处理数据集（支持离线）"""
    logger.info(f"加载数据集: {args.dataset_name}")
    
    # 判断是本地路径还是 HuggingFace 数据集名称
    if os.path.exists(args.dataset_name):
        # 从本地加载
        from datasets import load_from_disk
        dataset = load_from_disk(args.dataset_name)
        # 获取训练集
        if isinstance(dataset, dict):
            dataset = dataset['train']
    else:
        # 从 HuggingFace 加载
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    # ... 其余代码保持不变
```

### 5.3 启动离线训练

```bash
# 使用本地路径启动训练
python train_rlhf.py \
    --model_name=./offline_assets/models/Qwen--Qwen2.5-7B-Instruct \
    --dataset_name=./offline_assets/datasets/openbmb--UltraFeedback \
    --output_dir=./output_offline \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing

# 或使用多卡训练
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_rlhf.py \
    --model_name=./offline_assets/models/Qwen--Qwen2.5-7B-Instruct \
    --dataset_name=./offline_assets/datasets/openbmb--UltraFeedback \
    --output_dir=./output_offline \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing
```

---

## 📝 完整流程总结

### 在有网络的机器上

```bash
# 1. 下载资源
python download_assets.py

# 2. 下载依赖包
pip download -r requirements.txt -d offline_assets/wheels

# 3. 压缩（可选）
tar -czf offline_assets.tar.gz offline_assets/

# 4. 传输到开发机
scp offline_assets.tar.gz username@server:/path/to/project/
```

### 在开发机上

```bash
# 1. 解压（如果压缩了）
tar -xzf offline_assets.tar.gz

# 2. 安装依赖
bash install_offline.sh

# 3. 验证
python test_offline_assets.py

# 4. 启动训练
python train_rlhf.py \
    --model_name=./offline_assets/models/Qwen--Qwen2.5-7B-Instruct \
    --dataset_name=./offline_assets/datasets/openbmb--UltraFeedback \
    --output_dir=./output_offline
```

---

## 🔧 常见问题

### 1. 下载速度慢

**解决方案:**
```bash
# 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 使用 pip 国内镜像
pip download -r requirements.txt -d offline_assets/wheels \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 磁盘空间不足

**模型和数据集大小估算:**
- Qwen2.5-7B-Instruct: ~15GB
- UltraFeedback 数据集: ~2GB
- Python 依赖包: ~5GB
- **总计: ~22GB**

**解决方案:**
- 确保有足够的磁盘空间
- 可以只下载部分数据集

### 3. 传输中断

**使用 rsync 支持断点续传:**
```bash
rsync -avz --progress --partial \
    offline_assets/ \
    username@server:/path/to/project/offline_assets/
```

### 4. 依赖包版本不兼容

**解决方案:**
```bash
# 在有网络的机器上，使用与开发机相同的 Python 版本
python3.10 -m pip download -r requirements.txt -d offline_assets/wheels

# 或在开发机上手动安装特定版本
pip install --no-index --find-links=offline_assets/wheels torch==2.0.0
```

---

## 💡 最佳实践

1. **使用相同的 Python 版本**: 确保下载依赖包的机器和开发机使用相同的 Python 版本
2. **验证完整性**: 传输后验证文件完整性（使用 md5sum）
3. **保留下载的资源**: 下载的资源可以重复使用，建议保留备份
4. **分批传输**: 如果网络不稳定，可以分批传输（先模型，再数据集，最后依赖包）
5. **使用压缩**: 传输前压缩可以节省时间，但要确保开发机有足够空间解压

---

## 📚 相关文档

- `README.md` - 项目概述
- `DEPLOY_TO_SERVER.md` - 开发机部署指南
- `TRAINING_GUIDE.md` - 训练详细指南
- `QUICK_REFERENCE.md` - 快速参考

---

祝训练顺利！🚀
