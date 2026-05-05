# 开发机部署指南

## 📋 目录
1. [准备工作](#准备工作)
2. [上传代码到开发机](#上传代码到开发机)
3. [环境配置](#环境配置)
4. [运行训练](#运行训练)
5. [监控和管理](#监控和管理)
6. [常见问题](#常见问题)

---

## 1. 准备工作

### 开发机信息确认

在开始之前，确认你有以下信息：

```
开发机 IP: xxx.xxx.xxx.xxx
用户名: your_username
密码/SSH密钥: ******
GPU 数量: 4 或 8
```

### 本地准备

确保本地已经有完整的项目代码：

```bash
# 查看当前目录
pwd
# 应该在: D:\Dean-Fhx文件\北理\大学课题\rlhf-qwen3-baseline

# 查看文件列表
ls
```

---

## 2. 上传代码到开发机

### 方式 1: 使用 SCP 上传（推荐）

**从本地上传到开发机:**

```bash
# Windows PowerShell 或 Git Bash
# 上传整个项目目录
scp -r D:\Dean-Fhx文件\北理\大学课题\rlhf-qwen3-baseline username@server_ip:/home/username/

# 示例
scp -r . username@192.168.1.100:/home/username/rlhf-qwen3-baseline
```

**如果使用 SSH 密钥:**

```bash
scp -i /path/to/your/key.pem -r . username@server_ip:/home/username/rlhf-qwen3-baseline
```

### 方式 2: 使用 Git（如果开发机能访问 GitHub）

**在开发机上执行:**

```bash
# SSH 登录到开发机
ssh username@server_ip

# 克隆项目
git clone https://github.com/haixinfeng6-source/rlhf-qwen3-baseline.git
cd rlhf-qwen3-baseline
```

### 方式 3: 使用 rsync（Linux/Mac，更高效）

```bash
# 同步项目到开发机
rsync -avz --progress \
  --exclude='output*' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  . username@server_ip:/home/username/rlhf-qwen3-baseline/
```

### 方式 4: 使用 WinSCP（Windows 图形界面）

1. 下载安装 WinSCP: https://winscp.net/
2. 新建连接，填入服务器信息
3. 拖拽整个项目文件夹到服务器

---

## 3. 环境配置

### 3.1 登录开发机

```bash
ssh username@server_ip

# 或使用密钥
ssh -i /path/to/key.pem username@server_ip
```

### 3.2 检查 GPU

```bash
# 查看 GPU 信息
nvidia-smi

# 查看 CUDA 版本
nvcc --version
```

### 3.3 创建 Python 环境

**方式 A: 使用 conda（推荐）**

```bash
# 创建新环境
conda create -n rlhf python=3.10 -y

# 激活环境
conda activate rlhf

# 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**方式 B: 使用 venv**

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活环境
source venv/bin/activate

# 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3.4 安装项目依赖

```bash
# 进入项目目录
cd ~/rlhf-qwen3-baseline

# 安装依赖
pip install -r requirements.txt

# 如果下载慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.5 配置 HuggingFace 镜像（可选）

```bash
# 设置环境变量
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 或者临时设置
export HF_ENDPOINT=https://hf-mirror.com
```

### 3.6 测试环境

```bash
# 运行环境测试
python test_environment.py

# 运行快速测试
python run_local_test.py
```

---

## 4. 运行训练

### 4.1 配置训练参数

根据开发机的 GPU 配置，编辑 `config.yaml`:

```bash
# 使用 vim 或 nano 编辑
vim config.yaml

# 或者
nano config.yaml
```

**4 卡配置示例:**

```yaml
distributed:
  num_gpus: 4

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

**8 卡配置示例:**

```yaml
distributed:
  num_gpus: 8

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
```

### 4.2 启动训练

**方式 1: 使用启动脚本（推荐）**

```bash
# 修改脚本权限
chmod +x run_multi_gpu.sh

# 编辑脚本设置 GPU 数量
vim run_multi_gpu.sh
# 修改: NUM_GPUS=4 或 NUM_GPUS=8

# 启动训练
bash run_multi_gpu.sh
```

**方式 2: 使用 nohup 后台运行**

```bash
# 后台运行，输出到日志文件
nohup bash run_multi_gpu.sh > training.log 2>&1 &

# 查看进程
ps aux | grep python

# 查看日志
tail -f training.log
```

**方式 3: 使用 screen 或 tmux（推荐）**

```bash
# 使用 screen
screen -S rlhf_training
bash run_multi_gpu.sh

# 断开: Ctrl+A, D
# 重新连接: screen -r rlhf_training

# 使用 tmux
tmux new -s rlhf_training
bash run_multi_gpu.sh

# 断开: Ctrl+B, D
# 重新连接: tmux attach -t rlhf_training
```

**方式 4: 直接使用 torchrun**

```bash
# 4 卡训练
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_rlhf.py \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --dataset_name=openbmb/UltraFeedback \
    --output_dir=./output \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing
```

### 4.3 小规模测试（推荐先运行）

```bash
# 使用测试配置，只训练 50 条数据
python train_with_config.py --config config_local_test.yaml

# 预计 5-10 分钟完成
```

---

## 5. 监控和管理

### 5.1 监控 GPU 使用

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1

# 查看 GPU 利用率历史
nvidia-smi dmon -s u
```

### 5.2 监控训练进程

```bash
# 查看 Python 进程
ps aux | grep python

# 查看训练日志
tail -f training.log

# 查看 TensorBoard 日志
tail -f output/logs/events.out.tfevents.*
```

### 5.3 启动 TensorBoard

```bash
# 在开发机上启动 TensorBoard
tensorboard --logdir=./output/logs --port=6006 --host=0.0.0.0

# 在本地浏览器访问
http://server_ip:6006
```

**如果端口被防火墙阻止，使用 SSH 隧道:**

```bash
# 在本地执行
ssh -L 6006:localhost:6006 username@server_ip

# 然后在本地浏览器访问
http://localhost:6006
```

### 5.4 查看训练进度

```bash
# 查看输出目录
ls -lh output/

# 查看 checkpoint
ls -lh output/checkpoint-*/

# 查看日志文件
cat output/logs/training.log
```

### 5.5 停止训练

```bash
# 查找进程 PID
ps aux | grep train_rlhf.py

# 优雅停止（推荐）
kill -SIGTERM <PID>

# 强制停止
kill -9 <PID>

# 如果使用 screen/tmux
screen -r rlhf_training
# 然后按 Ctrl+C
```

---

## 6. 常见问题

### 6.1 上传文件失败

**问题:** Permission denied

**解决:**
```bash
# 在开发机上创建目录并设置权限
mkdir -p ~/rlhf-qwen3-baseline
chmod 755 ~/rlhf-qwen3-baseline

# 或使用 sudo
sudo chown -R username:username ~/rlhf-qwen3-baseline
```

### 6.2 CUDA Out of Memory

**解决方案:**

```bash
# 编辑配置文件
vim config.yaml

# 修改以下参数
quantization:
  use_4bit: true  # 使用 4-bit 量化

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # 增加梯度累积
  max_length: 256  # 减小序列长度
```

### 6.3 网络连接问题

**HuggingFace 下载慢或失败:**

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://proxy_server:port
export HTTPS_PROXY=http://proxy_server:port
```

### 6.4 多卡训练失败

**检查 NCCL:**

```bash
# 设置 NCCL 环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand

# 测试多卡通信
python -c "import torch; print(torch.cuda.device_count())"
```

### 6.5 SSH 连接断开导致训练中断

**使用 screen 或 tmux:**

```bash
# 安装 screen
sudo apt-get install screen

# 或安装 tmux
sudo apt-get install tmux

# 在 screen 中运行训练
screen -S training
bash run_multi_gpu.sh
```

### 6.6 磁盘空间不足

**检查磁盘空间:**

```bash
# 查看磁盘使用
df -h

# 查看目录大小
du -sh ~/rlhf-qwen3-baseline/*

# 清理缓存
rm -rf ~/.cache/huggingface/hub/*
```

---

## 7. 完整部署流程示例

```bash
# ========== 本地操作 ==========

# 1. 上传代码到开发机
scp -r . username@server_ip:/home/username/rlhf-qwen3-baseline


# ========== 开发机操作 ==========

# 2. SSH 登录
ssh username@server_ip

# 3. 进入项目目录
cd ~/rlhf-qwen3-baseline

# 4. 创建环境
conda create -n rlhf python=3.10 -y
conda activate rlhf

# 5. 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 6. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 7. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 8. 测试环境
python test_environment.py

# 9. 小规模测试
python train_with_config.py --config config_local_test.yaml

# 10. 启动完整训练（使用 tmux）
tmux new -s training
bash run_multi_gpu.sh

# 断开 tmux: Ctrl+B, D
# 退出 SSH
exit


# ========== 本地监控 ==========

# 11. SSH 隧道访问 TensorBoard
ssh -L 6006:localhost:6006 username@server_ip

# 12. 浏览器访问
# http://localhost:6006


# ========== 训练完成后 ==========

# 13. 下载模型到本地
scp -r username@server_ip:/home/username/rlhf-qwen3-baseline/output ./
```

---

## 8. 自动化部署脚本

创建一个自动化部署脚本 `deploy.sh`:

```bash
#!/bin/bash

# 配置信息
SERVER_IP="your_server_ip"
USERNAME="your_username"
REMOTE_DIR="/home/${USERNAME}/rlhf-qwen3-baseline"

echo "开始部署到开发机..."

# 1. 上传代码
echo "上传代码..."
rsync -avz --progress \
  --exclude='output*' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  . ${USERNAME}@${SERVER_IP}:${REMOTE_DIR}/

# 2. 远程执行安装命令
echo "配置环境..."
ssh ${USERNAME}@${SERVER_IP} << 'EOF'
cd ~/rlhf-qwen3-baseline
conda activate rlhf || conda create -n rlhf python=3.10 -y
conda activate rlhf
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python test_environment.py
EOF

echo "部署完成！"
echo "登录开发机: ssh ${USERNAME}@${SERVER_IP}"
echo "启动训练: cd ~/rlhf-qwen3-baseline && bash run_multi_gpu.sh"
```

使用方法:

```bash
# 修改脚本中的配置信息
vim deploy.sh

# 添加执行权限
chmod +x deploy.sh

# 运行部署
./deploy.sh
```

---

## 9. 训练完成后下载结果

```bash
# 下载训练好的模型
scp -r username@server_ip:/home/username/rlhf-qwen3-baseline/output ./

# 或使用 rsync
rsync -avz --progress \
  username@server_ip:/home/username/rlhf-qwen3-baseline/output/ \
  ./output/

# 只下载最终模型
scp -r username@server_ip:/home/username/rlhf-qwen3-baseline/output/final_model ./
```

---

## 10. 快速参考命令

```bash
# 上传代码
scp -r . user@server:/path/

# 登录服务器
ssh user@server

# 激活环境
conda activate rlhf

# 启动训练（后台）
tmux new -s training
bash run_multi_gpu.sh

# 查看 GPU
nvidia-smi

# 查看日志
tail -f training.log

# 下载结果
scp -r user@server:/path/output ./
```

---

祝训练顺利！🚀
