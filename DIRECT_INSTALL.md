# 直接安装指南（不使用虚拟环境）

适用于磁盘空间受限的情况。

## 快速开始

### 方法1: 使用自动脚本

```bash
# 拉取最新代码
git pull

# 运行直接安装脚本
bash install_direct.sh

# 选择CUDA版本（默认12.8）
# 脚本会自动安装所有依赖
```

### 方法2: 手动安装

```bash
# 1. 清理缓存
pip cache purge
rm -rf ~/.cache/pip

# 2. 安装PyTorch (CUDA 12.8)
pip install --no-cache-dir --user \
    torch==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 安装核心依赖
pip install --no-cache-dir --user \
    transformers==4.46.0 \
    datasets==3.1.0 \
    accelerate==1.0.0 \
    trl==0.11.4 \
    peft==0.13.2

# 4. 安装辅助依赖
pip install --no-cache-dir --user \
    numpy pandas tqdm PyYAML requests

# 5. 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from trl import PPOConfig; print('✅ TRL OK')"
```

## 关键参数说明

- `--no-cache-dir`: 不使用缓存，节省磁盘空间
- `--user`: 安装到用户目录 `~/.local/`，不需要root权限

## CUDA版本选择

### CUDA 12.8 (推荐)
```bash
pip install --no-cache-dir --user \
    torch==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### CUDA 12.1
```bash
pip install --no-cache-dir --user \
    torch==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8
```bash
pip install --no-cache-dir --user \
    torch==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only
```bash
pip install --no-cache-dir --user \
    torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

## 验证安装

```bash
# 完整验证
python << 'EOF'
import torch
import transformers
import trl

print("=" * 60)
print("环境验证")
print("=" * 60)
print(f"Python路径: {__import__('sys').executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"transformers: {transformers.__version__}")
print(f"TRL: {trl.__version__}")

try:
    from trl import PPOConfig, PPOTrainer
    from trl.models import AutoModelForCausalLMWithValueHead
    print("\n✅ 所有组件安装成功！")
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
EOF
```

## 运行训练

```bash
# 1. 下载离线资源
python download_assets.py

# 2. 运行训练
python train_rlhf_simple.py --config config_cluster.yaml

# 3. 或提交SLURM作业
sbatch submit_train.slurm
```

## 节省空间的技巧

### 1. 清理缓存
```bash
# pip缓存
pip cache purge

# 通用缓存
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch

# 临时文件
rm -rf /tmp/$USER/*
```

### 2. 使用临时目录
```bash
# 如果其他磁盘有空间，设置临时目录
export TMPDIR=/data/$USER/tmp
mkdir -p $TMPDIR

# 然后运行安装
bash install_direct.sh
```

### 3. 只安装必要的包
```bash
# 最小安装（只安装核心包）
pip install --no-cache-dir --user torch transformers trl

# 其他包按需安装
```

### 4. 检查已安装包
```bash
# 查看安装位置
pip list --user

# 查看大小
du -sh ~/.local/lib/python*/site-packages
```

## 包安装位置

使用`--user`安装时，包会安装到：
- **Python包**: `~/.local/lib/python3.X/site-packages/`
- **可执行文件**: `~/.local/bin/`

## 卸载包

```bash
# 卸载单个包
pip uninstall package_name

# 卸载所有用户包（谨慎！）
pip freeze --user | xargs pip uninstall -y
```

## 常见问题

### Q1: 权限错误
```bash
# 使用--user参数，不需要root权限
pip install --user package_name
```

### Q2: 找不到包
```bash
# 确保添加--user路径到PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python3.X/site-packages:$PYTHONPATH
```

### Q3: 版本冲突
```bash
# 升级pip
python -m pip install --upgrade pip --user

# 强制重新安装
pip install --force-reinstall --user package_name
```

### Q4: 磁盘仍然不足
```bash
# 清理更多空间
du -sh ~/* | sort -hr | head -10

# 删除不需要的文件
# 或联系管理员增加配额
```

## 优势

✅ **节省空间**: 不需要虚拟环境的重复文件  
✅ **简单直接**: 不需要激活/停用环境  
✅ **系统级共享**: 可以使用已安装的系统包  
✅ **无需root**: 使用--user安装即可  

## 劣势

⚠️ **版本冲突**: 可能与系统包冲突  
⚠️ **污染环境**: 影响其他Python项目  
⚠️ **难以隔离**: 不容易切换不同版本  

## 建议

- 如果磁盘空间充足，推荐使用虚拟环境
- 如果空间受限，使用直接安装
- 可以在集群上直接安装（因为集群环境通常比较干净）

## 从虚拟环境迁移

如果之前创建了虚拟环境并想删除：

```bash
# 1. 退出虚拟环境（如果激活了）
deactivate

# 2. 删除虚拟环境
rm -rf rlhf-env

# 3. 直接安装
bash install_direct.sh
```

## 完整示例

```bash
# 完整的安装流程
cd ~/rlhf-qwen3-baseline
git pull
bash install_direct.sh

# 验证
python -c "from trl import PPOConfig; print('✅ OK')"

# 下载资源
python download_assets.py

# 测试
python train_rlhf_simple.py --max_samples=5

# 训练
sbatch submit_train.slurm
```