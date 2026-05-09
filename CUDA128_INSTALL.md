# CUDA 12.8 快速安装指南

## 检查CUDA版本

```bash
# 检查NVIDIA驱动和CUDA版本
nvidia-smi

# 输出示例:
# CUDA Version: 12.8
```

## 安装步骤

### 方法1: 使用自动脚本（推荐）

```bash
# 运行自动配置脚本
bash setup_cpu_env.sh

# 选择 3) CUDA 12.8（默认选项）
```

### 方法2: 手动安装

```bash
# 1. 创建虚拟环境
python3.10 -m venv rlhf-env
source rlhf-env/bin/activate

# 2. 安装PyTorch (CUDA 12.8)
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 3. 安装其他依赖
pip install transformers==4.46.0
pip install trl==0.11.4
pip install datasets==3.1.0
pip install accelerate==1.0.0
pip install peft==0.13.2
pip install numpy pandas tqdm PyYAML requests

# 4. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from trl import PPOConfig, PPOTrainer; print('✅ TRL OK')"
```

### 方法3: 使用requirements文件

```bash
# 1. 先安装PyTorch
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. 然后安装其他依赖
pip install -r requirements_cpu_cluster.txt --no-deps

# 注意: --no-deps 避免重新安装torch
```

## 版本对应关系

| CUDA版本 | PyTorch版本 | 安装命令 |
|---------|------------|----------|
| 11.8 | 2.1.0+cu118 | `pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | 2.5.1+cu121 | `pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| **12.8** | **2.7.0+cu128** | `pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128` |
| CPU | 2.7.0+cpu | `pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu` |

## 验证清单

```bash
# 1. 检查PyTorch和CUDA
python << EOF
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

# 2. 检查TRL
python -c "from trl import PPOConfig, PPOTrainer; print('✅ TRL导入成功')"

# 3. 检查transformers
python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# 4. 检查其他包
python << EOF
import datasets
import peft
import accelerate
print(f"datasets: {datasets.__version__}")
print(f"peft: {peft.__version__}")
print(f"accelerate: {accelerate.__version__}")
EOF
```

## 常见问题

### Q1: 找不到torch版本

**错误**: `ERROR: No matching distribution found for torch==2.7.0+cu128`

**解决**:
```bash
# 确保使用正确的index-url
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 或使用--extra-index-url
pip install torch==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

### Q2: CUDA不可用

**检查**:
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA库
ldconfig -p | grep cuda

# 如果CUDA不可用，使用CPU版本
pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Q3: 版本冲突

**解决**:
```bash
# 清理环境重新安装
pip uninstall torch transformers trl datasets accelerate peft -y

# 按顺序重新安装
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.46.0
pip install trl==0.11.4
pip install datasets==3.1.0
pip install accelerate==1.0.0
pip install peft==0.13.2
```

### Q4: TRL导入错误

**解决**:
```bash
# TRL 0.11.4 支持 transformers 4.46.0
pip install transformers==4.46.0 --force-reinstall
pip install trl==0.11.4 --force-reinstall

# 测试
python -c "from trl import PPOConfig, PPOTrainer; print('✅ OK')"
```

## 下一步

安装完成后：

1. **下载离线资源**:
   ```bash
   python download_assets.py
   ```

2. **运行测试**:
   ```bash
   python train_rlhf_simple.py --config config_cluster.yaml
   ```

3. **打包环境**（如需传输到集群）:
   ```bash
   tar -czf rlhf-env.tar.gz rlhf-env/
   tar -czf rlhf-project.tar.gz rlhf-qwen3-baseline/
   ```

## 参考链接

- [PyTorch安装指南](https://pytorch.org/get-started/locally/)
- [PyTorch CUDA兼容性](https://pytorch.org/get-started/previous-versions/)
- [TRL文档](https://huggingface.co/docs/trl)
- [Transformers文档](https://huggingface.co/docs/transformers)