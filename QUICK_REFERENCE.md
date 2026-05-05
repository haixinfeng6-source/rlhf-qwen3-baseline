# 快速参考卡片

## 🚀 部署到开发机

### 方式 1: 自动化脚本（推荐）

**Linux/Mac:**
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

### 方式 2: 手动上传

```bash
# 上传代码
scp -r . username@server_ip:/home/username/rlhf-qwen3-baseline

# 或使用 rsync
rsync -avz --exclude='output*' --exclude='.git' \
  . username@server_ip:/home/username/rlhf-qwen3-baseline/
```

---

## 🔧 开发机环境配置

```bash
# 1. SSH 登录
ssh username@server_ip

# 2. 进入目录
cd ~/rlhf-qwen3-baseline

# 3. 创建环境
conda create -n rlhf python=3.10 -y
conda activate rlhf

# 4. 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 5. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 7. 测试环境
python test_environment.py
```

---

## 🎯 启动训练

### 小规模测试（推荐先运行）

```bash
python train_with_config.py --config config_local_test.yaml
```

### 完整训练

**使用 tmux（推荐）:**
```bash
# 创建 tmux 会话
tmux new -s training

# 启动训练
bash run_multi_gpu.sh

# 断开: Ctrl+B, D
# 重连: tmux attach -t training
```

**使用 screen:**
```bash
# 创建 screen 会话
screen -S training

# 启动训练
bash run_multi_gpu.sh

# 断开: Ctrl+A, D
# 重连: screen -r training
```

**后台运行:**
```bash
nohup bash run_multi_gpu.sh > training.log 2>&1 &
```

---

## 📊 监控训练

### GPU 监控

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

### 日志查看

```bash
# 查看训练日志
tail -f training.log

# 查看最近 100 行
tail -n 100 training.log

# 搜索错误
grep -i error training.log
```

### TensorBoard

```bash
# 在开发机启动
tensorboard --logdir=./output/logs --port=6006 --host=0.0.0.0

# 本地访问（SSH 隧道）
ssh -L 6006:localhost:6006 username@server_ip
# 浏览器访问: http://localhost:6006
```

---

## 🛠️ 常用命令

### 进程管理

```bash
# 查看训练进程
ps aux | grep train_rlhf

# 停止训练
kill -SIGTERM <PID>

# 强制停止
kill -9 <PID>
```

### 文件管理

```bash
# 查看输出目录
ls -lh output/

# 查看 checkpoint
ls -lh output/checkpoint-*/

# 查看磁盘空间
df -h
du -sh output/
```

### 环境管理

```bash
# 激活环境
conda activate rlhf

# 查看已安装包
pip list

# 更新包
pip install --upgrade transformers
```

---

## 📥 下载结果

### 下载训练好的模型

```bash
# 下载整个输出目录
scp -r username@server_ip:/home/username/rlhf-qwen3-baseline/output ./

# 只下载最终模型
scp -r username@server_ip:/home/username/rlhf-qwen3-baseline/output/final_model ./

# 使用 rsync（更高效）
rsync -avz --progress \
  username@server_ip:/home/username/rlhf-qwen3-baseline/output/ \
  ./output/
```

---

## ⚙️ 配置调整

### 显存不足

编辑 `config.yaml`:

```yaml
quantization:
  use_4bit: true  # 改为 4-bit

lora:
  r: 8  # 减小 rank

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # 增加
  max_length: 256  # 减小
```

### 加速训练

```yaml
training:
  per_device_train_batch_size: 2  # 增加（如果显存允许）
  gradient_accumulation_steps: 4  # 减少
  gradient_checkpointing: false  # 关闭（如果显存充足）

ppo:
  epochs: 2  # 减少 PPO epochs
```

### 多卡配置

```yaml
distributed:
  num_gpus: 4  # 或 8

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # 4卡
  # gradient_accumulation_steps: 4  # 8卡
```

---

## 🐛 故障排查

### CUDA Out of Memory

```bash
# 1. 检查 GPU 使用
nvidia-smi

# 2. 杀死其他进程
kill -9 <PID>

# 3. 清理缓存
python -c "import torch; torch.cuda.empty_cache()"

# 4. 调整配置（见上方"显存不足"）
```

### 网络问题

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 设置代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 测试连接
curl -I https://huggingface.co
```

### 多卡训练失败

```bash
# 设置 NCCL 环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# 测试多卡
python -c "import torch; print(torch.cuda.device_count())"

# 检查端口
netstat -tuln | grep 29500
```

---

## 📝 文件说明

| 文件 | 说明 |
|------|------|
| `train_rlhf.py` | PPO 训练脚本 |
| `train_grpo.py` | GRPO 训练脚本 |
| `train_with_config.py` | 配置文件训练 |
| `config.yaml` | 训练配置 |
| `config_local_test.yaml` | 测试配置 |
| `run_multi_gpu.sh` | 多卡启动脚本 |
| `deploy.sh` | 自动部署脚本 |
| `test_environment.py` | 环境测试 |
| `evaluate_model.py` | 模型评估 |

---

## 🔗 相关文档

- `README.md` - 项目概述
- `TRAINING_GUIDE.md` - 详细训练指南
- `LOCAL_SETUP.md` - 本地运行指南
- `DEPLOY_TO_SERVER.md` - 开发机部署详细指南

---

## 💡 最佳实践

1. ✅ 先运行小规模测试验证环境
2. ✅ 使用 tmux/screen 防止 SSH 断开
3. ✅ 定期检查 GPU 利用率
4. ✅ 保存多个 checkpoint
5. ✅ 使用 TensorBoard 监控训练
6. ✅ 定期备份重要 checkpoint

---

## 📞 获取帮助

- GitHub: https://github.com/haixinfeng6-source/rlhf-qwen3-baseline
- Issues: https://github.com/haixinfeng6-source/rlhf-qwen3-baseline/issues
