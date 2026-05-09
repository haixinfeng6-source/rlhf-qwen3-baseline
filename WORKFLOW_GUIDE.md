# RLHF 项目开发工作流程指南

## 开发流程概览

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌─────────────┐
│  本地开发    │─────▶│   GitHub     │─────▶│  远程CPU    │─────▶│  集群GPU    │
│  (Windows)  │      │   (代码库)    │      │  (联网配置) │      │  (离线训练) │
└─────────────┘      └──────────────┘      └─────────────┘      └─────────────┘
      │                                            │                    │
      │                                            │                    │
      ▼                                            ▼                    ▼
  代码修改                                      环境配置              实际训练
  功能测试                                      依赖下载              模型训练
  文档更新                                      离线资源准备           结果输出
```

## 详细步骤

### 步骤1: 本地开发（Windows）

#### 1.1 环境准备
```bash
# 本地不需要完整的训练环境，只需要基本的Python用于代码修改
# 可选：创建轻量级虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
pip install pyyaml requests  # 仅用于测试配置文件和下载脚本
```

#### 1.2 代码修改
- 修改训练代码 `train_rlhf.py`
- 更新配置文件 `config.yaml`
- 测试配置文件语法
- 编写/更新文档

#### 1.3 本地验证
```bash
# 验证YAML配置文件
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# 验证Python语法
python -m py_compile train_rlhf.py

# 测试下载脚本（可选）
python download_assets.py --test
```

#### 1.4 提交到GitHub
```bash
git add .
git commit -m "描述修改内容"
git push
```

### 步骤2: 远程CPU配置（联网）

#### 2.1 环境准备

**重要**: 在CPU节点创建一个干净、可移植的环境

```bash
# 登录CPU节点
ssh username@cpu-node

# 克隆代码
git clone https://github.com/your-repo/rlhf-qwen3-baseline
cd rlhf-qwen3-baseline

# 创建虚拟环境（关键：使用--system-site-packages以继承系统包）
python3.10 -m venv --system-site-packages rlhf-env
source rlhf-env/bin/activate

# 或者使用conda（推荐，更容易移植）
conda create -n rlhf python=3.10 -y
conda activate rlhf
```

#### 2.2 安装依赖（核心步骤）

**关键**: 使用固定的、经过验证的版本

```bash
# 方案A: 使用精简版依赖（推荐）
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install trl==0.11.0
pip install datasets==2.14.0
pip install accelerate==0.25.0
pip install peft==0.7.0

# 验证安装
python -c "from trl import PPOConfig, PPOTrainer; print('✅ TRL导入成功')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 方案B: 使用requirements文件
pip install -r requirements_cpu.txt
```

#### 2.3 下载离线资源

```bash
# 下载模型和数据集到项目目录
python download_assets.py

# 验证下载
ls -lh offline_assets/models/Qwen--Qwen3-8B/
ls -lh offline_assets/datasets/openbmb--UltraFeedback/

# 下载Python包（用于集群离线安装）
pip download -r requirements_cpu.txt -d offline_assets/wheels/
```

#### 2.4 CPU节点测试

```bash
# 使用CPU模式测试（小规模）
python train_rlhf.py \
  --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
  --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
  --output_dir="./test_cpu" \
  --max_samples=10 \
  --device=cpu

# 验证结果
ls -lh test_cpu/
```

#### 2.5 打包环境（关键步骤）

**方法1: 导出虚拟环境（简单）**
```bash
# 导出环境配置
pip freeze > requirements_exact.txt
conda env export > environment.yml  # 如果用conda

# 打包整个项目
cd ..
tar -czf rlhf-project.tar.gz rlhf-qwen3-baseline/
```

**方法2: 打包conda环境（推荐，可移植性更好）**
```bash
# 打包conda环境
conda pack -n rlhf -o rlhf_env.tar.gz

# 或使用virtualenv的relocatable模式
virtualenv-clone rlhf-env rlhf-env-portable
```

### 步骤3: 集群GPU部署（离线）

#### 3.1 上传到集群

```bash
# 从CPU节点传输到集群
scp rlhf-project.tar.gz username@cluster-node:~/
scp rlhf_env.tar.gz username@cluster-node:~/

# 或使用rsync（推荐）
rsync -avz --progress rlhf-qwen3-baseline/ username@cluster-node:~/rlhf-qwen3-baseline/
```

#### 3.2 在集群上解压和配置

```bash
# 登录集群
ssh username@cluster-node

# 解压项目
cd ~
tar -xzf rlhf-project.tar.gz
cd rlhf-qwen3-baseline

# 解压环境（如果打包了）
mkdir -p rlhf-env
tar -xzf rlhf_env.tar.gz -C rlhf-env/

# 激活环境
source rlhf-env/bin/activate

# 或使用conda-pack的环境
source rlhf-env/bin/activate
```

#### 3.3 集群环境验证

```bash
# 检查环境
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# 测试TRL导入
python -c "from trl import PPOConfig, PPOTrainer; print('✅ TRL OK')"

# 测试模型加载
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./offline_assets/models/Qwen--Qwen3-8B', trust_remote_code=True)
print(f'Tokenizer: {len(tokenizer)} vocab')
"
```

#### 3.4 提交训练作业

```bash
# 方案A: 直接运行（交互式）
python train_rlhf.py \
  --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
  --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback" \
  --output_dir="./output_train" \
  --max_samples=100

# 方案B: 使用SLURM作业脚本
sbatch submit_train.slurm
```

## 关键配置文件

### requirements_cpu.txt（精简版，用于CPU和集群）
```txt
# PyTorch（根据CUDA版本选择）
torch==2.1.0

# HuggingFace核心
transformers==4.36.0
datasets==2.14.0
accelerate==0.25.0

# RLHF训练
trl==0.11.0

# 参数高效微调
peft==0.7.0

# 工具包（可选）
tqdm
tensorboard
```

### config_cluster.yaml（集群专用配置）
```yaml
# 模型配置
model:
  name: "./offline_assets/models/Qwen--Qwen3-8B"
  trust_remote_code: true

# 数据集配置
dataset:
  name: "./offline_assets/datasets/openbmb--UltraFeedback"
  split: "train"
  max_samples: 1000  # 根据实际调整

# LoRA配置
lora:
  enabled: true
  r: 8  # 减小以节省显存
  alpha: 32
  dropout: 0.05

# 训练配置
training:
  output_dir: "./output_cluster"
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-5
  max_length: 256  # 减小以节省显存
  gradient_checkpointing: true
  fp16: true

# 分布式配置
distributed:
  backend: "nccl"
  num_gpus: 4

# 离线模式
offline:
  enabled: true
```

### submit_train.slurm（SLURM作业脚本）
```bash
#!/bin/bash
#SBATCH --job-name=rlhf-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=rlhf-%j.out
#SBATCH --error=rlhf-%j.err

# 加载模块（根据集群配置）
module load cuda/11.8
module load python/3.10

# 激活环境
source ~/rlhf-env/bin/activate

# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 运行训练
cd ~/rlhf-qwen3-baseline

python train_rlhf.py \
  --config config_cluster.yaml \
  --model_name="./offline_assets/models/Qwen--Qwen3-8B" \
  --dataset_name="./offline_assets/datasets/openbmb--UltraFeedback"

# 记录结束时间
echo "Training finished at $(date)"
```

## 常见问题解决

### 问题1: CPU和集群环境不一致
**解决**: 使用conda-pack或virtualenv-clone打包整个环境

### 问题2: 离线资源缺失
**解决**: 在CPU节点完整下载，验证后再传输

### 问题3: 集群上导入错误
**解决**: 
1. 确保环境完全复制
2. 检查PYTHONPATH设置
3. 验证所有依赖版本

### 问题4: CUDA版本不匹配
**解决**: 
1. 检查集群CUDA版本: `nvidia-smi`
2. 安装对应的PyTorch版本
3. 在CPU节点安装时指定正确的CUDA版本

## 验证清单

### CPU节点验证 ✓
- [ ] 虚拟环境创建成功
- [ ] 所有依赖安装成功
- [ ] TRL可以正常导入
- [ ] 离线资源下载完整
- [ ] CPU测试运行成功
- [ ] 环境打包成功

### 集群验证 ✓
- [ ] 项目文件传输完整
- [ ] 环境解压成功
- [ ] GPU可用且数量正确
- [ ] TRL导入正常
- [ ] 模型可以加载
- [ ] 数据集可以加载
- [ ] 小规模测试成功

## 最佳实践

1. **版本固定**: 所有依赖都使用固定版本号
2. **环境隔离**: 使用虚拟环境避免冲突
3. **分步验证**: 每一步都进行验证
4. **文档记录**: 记录所有操作和遇到的问题
5. **备份**: 定期备份代码和环境配置

## 紧急恢复

如果集群上出现问题：

```bash
# 1. 检查环境
python -c "import sys; print(sys.path)"
pip list

# 2. 重新安装依赖（如果有网络）
pip install --no-index --find-links=offline_assets/wheels/ -r requirements_cpu.txt

# 3. 从CPU节点重新传输
# 在CPU节点执行:
rsync -avz --delete rlhf-qwen3-baseline/ username@cluster-node:~/rlhf-qwen3-baseline/
```

## 联系支持

如果遇到无法解决的问题：
1. 记录完整错误信息
2. 保存环境配置 (`pip freeze > env.txt`)
3. 描述复现步骤
4. 联系集群管理员或在GitHub提issue