#!/bin/bash

# ========================================
# 自动化部署脚本 - 部署到开发机
# ========================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置信息（请修改为你的实际信息）
SERVER_IP="your_server_ip"          # 例如: 192.168.1.100
USERNAME="your_username"             # 例如: ubuntu
REMOTE_DIR="/home/${USERNAME}/rlhf-qwen3-baseline"
SSH_KEY=""                           # 如果使用 SSH 密钥，填写路径

# 函数：打印信息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：检查配置
check_config() {
    print_info "检查配置..."
    
    if [ "$SERVER_IP" = "your_server_ip" ]; then
        print_error "请先修改脚本中的 SERVER_IP"
        exit 1
    fi
    
    if [ "$USERNAME" = "your_username" ]; then
        print_error "请先修改脚本中的 USERNAME"
        exit 1
    fi
    
    print_info "配置检查通过"
}

# 函数：测试 SSH 连接
test_ssh() {
    print_info "测试 SSH 连接..."
    
    if [ -n "$SSH_KEY" ]; then
        ssh -i "$SSH_KEY" -o ConnectTimeout=5 ${USERNAME}@${SERVER_IP} "echo 'SSH 连接成功'" 2>/dev/null
    else
        ssh -o ConnectTimeout=5 ${USERNAME}@${SERVER_IP} "echo 'SSH 连接成功'" 2>/dev/null
    fi
    
    if [ $? -eq 0 ]; then
        print_info "SSH 连接测试成功"
        return 0
    else
        print_error "SSH 连接失败，请检查服务器地址、用户名和密码/密钥"
        return 1
    fi
}

# 函数：上传代码
upload_code() {
    print_info "上传代码到开发机..."
    
    # 构建 rsync 命令
    RSYNC_CMD="rsync -avz --progress \
        --exclude='output*' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='*.log'"
    
    if [ -n "$SSH_KEY" ]; then
        RSYNC_CMD="$RSYNC_CMD -e 'ssh -i $SSH_KEY'"
    fi
    
    RSYNC_CMD="$RSYNC_CMD . ${USERNAME}@${SERVER_IP}:${REMOTE_DIR}/"
    
    eval $RSYNC_CMD
    
    if [ $? -eq 0 ]; then
        print_info "代码上传成功"
        return 0
    else
        print_error "代码上传失败"
        return 1
    fi
}

# 函数：配置环境
setup_environment() {
    print_info "配置开发机环境..."
    
    SSH_CMD="ssh"
    if [ -n "$SSH_KEY" ]; then
        SSH_CMD="ssh -i $SSH_KEY"
    fi
    
    $SSH_CMD ${USERNAME}@${SERVER_IP} << 'ENDSSH'
        set -e
        
        echo "进入项目目录..."
        cd ~/rlhf-qwen3-baseline
        
        echo "检查 conda..."
        if command -v conda &> /dev/null; then
            echo "Conda 已安装"
            
            # 检查环境是否存在
            if conda env list | grep -q "^rlhf "; then
                echo "环境 rlhf 已存在，激活环境..."
                source $(conda info --base)/etc/profile.d/conda.sh
                conda activate rlhf
            else
                echo "创建新环境 rlhf..."
                conda create -n rlhf python=3.10 -y
                source $(conda info --base)/etc/profile.d/conda.sh
                conda activate rlhf
                
                echo "安装 PyTorch..."
                conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
            fi
        else
            echo "Conda 未安装，使用 venv..."
            if [ ! -d "venv" ]; then
                python3 -m venv venv
            fi
            source venv/bin/activate
        fi
        
        echo "安装项目依赖..."
        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        
        echo "设置 HuggingFace 镜像..."
        echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
        
        echo "测试环境..."
        python test_environment.py
        
        echo "环境配置完成！"
ENDSSH
    
    if [ $? -eq 0 ]; then
        print_info "环境配置成功"
        return 0
    else
        print_error "环境配置失败"
        return 1
    fi
}

# 函数：显示后续步骤
show_next_steps() {
    echo ""
    echo "=========================================="
    echo "部署完成！后续步骤："
    echo "=========================================="
    echo ""
    echo "1. 登录开发机："
    if [ -n "$SSH_KEY" ]; then
        echo "   ssh -i $SSH_KEY ${USERNAME}@${SERVER_IP}"
    else
        echo "   ssh ${USERNAME}@${SERVER_IP}"
    fi
    echo ""
    echo "2. 进入项目目录："
    echo "   cd ~/rlhf-qwen3-baseline"
    echo ""
    echo "3. 激活环境："
    echo "   conda activate rlhf"
    echo ""
    echo "4. 小规模测试（推荐）："
    echo "   python train_with_config.py --config config_local_test.yaml"
    echo ""
    echo "5. 启动完整训练："
    echo "   tmux new -s training"
    echo "   bash run_multi_gpu.sh"
    echo ""
    echo "6. 监控训练："
    echo "   - 查看 GPU: nvidia-smi"
    echo "   - 查看日志: tail -f training.log"
    echo "   - TensorBoard: tensorboard --logdir=./output/logs --host=0.0.0.0"
    echo ""
    echo "=========================================="
}

# 主函数
main() {
    echo "=========================================="
    echo "RLHF 项目自动化部署脚本"
    echo "=========================================="
    echo ""
    
    # 检查配置
    check_config
    
    # 测试连接
    if ! test_ssh; then
        exit 1
    fi
    
    # 上传代码
    if ! upload_code; then
        exit 1
    fi
    
    # 配置环境
    read -p "是否配置开发机环境？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if ! setup_environment; then
            print_warning "环境配置失败，但代码已上传"
        fi
    else
        print_info "跳过环境配置"
    fi
    
    # 显示后续步骤
    show_next_steps
}

# 运行主函数
main
