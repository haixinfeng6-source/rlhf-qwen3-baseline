# RLHF Qwen3-8B - 快速开始指南

## 🎯 项目目标

基于Qwen3-8B模型和UltraFeedback数据集，使用PPO算法进行RLHF训练。

## 📋 开发工作流程

```
本地开发 → GitHub → 远程CPU配置 → 集群GPU训练
```

## 🚀 快速开始

### 1. 本地开发（Windows）

```bash
# 克隆代码
git clone https://github.com/your-repo/rlhf-qwen3-baseline
cd rlhf-qwen3-baseline

# 修改代码和配置
# ...

# 提交到GitHub
git add .
git commit -m "更新内容"
git push
```

### 2. CPU节点配置（联网）

```bash
# 登录CPU节点
ssh username@cpu-node

# 克隆代码
git clone https://github.com/your-repo/rlhf-qwen3-baseline
cd rlhf-qwen3-baseline

# 创建环境
python3.10 -m venv rlhf-env
source rlhf-env/bin/activate

# 安装依赖
pip install -r requirements_cpu_cluster.txt

# 验证安装
python -c "from trl import PPOConfig, PPOTrainer; print('✅ TRL OK')"

# 下载离线资源
python download_assets.py

# 测试（CPU模式）
python train_rlhf_simple.py --config config_cluster.yaml
```

### 3. 集群GPU训练（离线）

```bash
# 从CPU节点传输到集群
rsync -avz rlhf-qwen3-baseline/ username@cluster-node:~/rlhf-qwen3-baseline/

# 登录集群
ssh username@cluster-node

# 进入项目
cd ~/rlhf-qwen3-baseline

# 激活环境（如果打包了）
source rlhf-env/bin/activate

# 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 提交作业
sbatch submit_train.slurm

# 或直接运行
python train_rlhf_simple.py --config config_cluster.yaml
```

## 📁 项目结构

```
rlhf-qwen3-baseline/
├── train_rlhf_simple.py      # 简化训练脚本（推荐使用）
├── config_cluster.yaml       # 集群配置文件
├── requirements_cpu_cluster.txt  # 精简依赖
├── download_assets.py        # 下载离线资源
├── submit_train.slurm        # SLURM作业脚本
├── offline_assets/           # 离线资源
│   ├── models/              # 模型文件
│   ├── datasets/            # 数据集
│   └── wheels/              # Python包
└── WORKFLOW_GUIDE.md        # 详细工作流程指南
```

## 🔧 核心文件说明

### requirements_cpu_cluster.txt
精简版依赖，只包含必要包，避免版本冲突。

### train_rlhf_simple.py
简化训练脚本，特点：
- 最小化依赖
- 详细错误处理
- 支持CPU和GPU
- 离线资源加载

### config_cluster.yaml
集群专用配置，包含模型、数据集、训练参数。

### submit_train.slurm
SLURM作业脚本，包含资源申请和环境配置。

## ⚙️ 配置文件

### config_cluster.yaml 示例

```yaml
model:
  name: "./offline_assets/models/Qwen--Qwen3-8B"
  trust_remote_code: true

dataset:
  name: "./offline_assets/datasets/openbmb--UltraFeedback"
  split: "train"
  max_samples: 1000

lora:
  enabled: true
  r: 8
  alpha: 32
  dropout: 0.05

training:
  output_dir: "./output_cluster"
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-5
  max_length: 256
  gradient_checkpointing: true
  fp16: true
```

## 🐛 常见问题

### Q1: TRL导入失败
```bash
# 安装固定版本
pip install trl==0.11.0 --force-reinstall
```

### Q2: bitsandbytes错误
```bash
# 不使用bitsandbytes，使用train_rlhf_simple.py
# 它自动禁用量化，使用FP16替代
```

### Q3: CUDA不可用
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应的PyTorch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### Q4: 离线资源加载失败
```bash
# 确保使用绝对路径或正确的相对路径
# 确保文件权限正确
ls -la offline_assets/models/Qwen--Qwen3-8B/
```

## 📊 性能优化

### 显存优化
- LoRA rank: 8（减小）
- 序列长度: 256（减小）
- 梯度累积: 16（增加）
- 批次大小: 1（最小）
- 梯度检查点: 启用

### 训练速度
- max_samples: 先用小数据测试
- num_train_epochs: 1（测试）
- logging_steps: 50（减少日志）

## 📝 详细文档

- `WORKFLOW_GUIDE.md` - 完整工作流程指南
- `README.md` - 项目总览
- `CLUSTER_SETUP_SIMPLE.md` - 集群配置指南

## ✅ 验证清单

### CPU节点
- [ ] 环境创建成功
- [ ] 依赖安装成功
- [ ] TRL可以导入
- [ ] 离线资源下载完整
- [ ] CPU测试成功

### 集群节点
- [ ] 项目文件传输完整
- [ ] 环境可用
- [ ] GPU可用
- [ ] 模型可以加载
- [ ] 数据集可以加载
- [ ] 小规模测试成功

## 🆘 获取帮助

遇到问题时：

1. 查看错误日志
2. 运行环境检查: `python train_rlhf_simple.py`
3. 查看详细文档: `WORKFLOW_GUIDE.md`
4. 提issue: GitHub Issues

## 📄 License

MIT