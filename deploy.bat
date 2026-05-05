@echo off
REM ========================================
REM 自动化部署脚本 - Windows 版本
REM ========================================

setlocal enabledelayedexpansion

REM 配置信息（请修改为你的实际信息）
set SERVER_IP=your_server_ip
set USERNAME=your_username
set REMOTE_DIR=/home/%USERNAME%/rlhf-qwen3-baseline

echo ==========================================
echo RLHF 项目自动化部署脚本 (Windows)
echo ==========================================
echo.

REM 检查配置
if "%SERVER_IP%"=="your_server_ip" (
    echo [ERROR] 请先修改脚本中的 SERVER_IP
    pause
    exit /b 1
)

if "%USERNAME%"=="your_username" (
    echo [ERROR] 请先修改脚本中的 USERNAME
    pause
    exit /b 1
)

echo [INFO] 配置检查通过
echo.

REM 检查 SSH 是否可用
where ssh >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 SSH 命令
    echo 请安装 OpenSSH 或使用 Git Bash
    pause
    exit /b 1
)

REM 测试 SSH 连接
echo [INFO] 测试 SSH 连接...
ssh -o ConnectTimeout=5 %USERNAME%@%SERVER_IP% "echo SSH 连接成功" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] SSH 连接失败
    echo 请检查服务器地址、用户名和密码
    pause
    exit /b 1
)
echo [INFO] SSH 连接测试成功
echo.

REM 上传代码
echo [INFO] 上传代码到开发机...
echo 这可能需要几分钟，请耐心等待...
echo.

REM 使用 scp 上传（排除不需要的文件）
scp -r ^
    -o "SendEnv=LC_*" ^
    *.py *.sh *.bat *.yaml *.yml *.txt *.md ^
    %USERNAME%@%SERVER_IP%:%REMOTE_DIR%/

if %errorlevel% neq 0 (
    echo [ERROR] 代码上传失败
    pause
    exit /b 1
)

echo [INFO] 代码上传成功
echo.

REM 询问是否配置环境
set /p SETUP_ENV="是否配置开发机环境？(y/n): "
if /i not "%SETUP_ENV%"=="y" (
    echo [INFO] 跳过环境配置
    goto :show_next_steps
)

REM 配置环境
echo [INFO] 配置开发机环境...
ssh %USERNAME%@%SERVER_IP% "bash -s" << 'EOF'
    set -e
    
    echo "进入项目目录..."
    cd ~/rlhf-qwen3-baseline
    
    echo "检查 conda..."
    if command -v conda &> /dev/null; then
        echo "Conda 已安装"
        
        if conda env list | grep -q "^rlhf "; then
            echo "环境 rlhf 已存在"
            source $(conda info --base)/etc/profile.d/conda.sh
            conda activate rlhf
        else
            echo "创建新环境 rlhf..."
            conda create -n rlhf python=3.10 -y
            source $(conda info --base)/etc/profile.d/conda.sh
            conda activate rlhf
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        fi
    else
        echo "使用 venv..."
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
    fi
    
    echo "安装依赖..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    echo "设置镜像..."
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    
    echo "测试环境..."
    python test_environment.py
    
    echo "环境配置完成！"
EOF

if %errorlevel% neq 0 (
    echo [WARNING] 环境配置失败，但代码已上传
) else (
    echo [INFO] 环境配置成功
)
echo.

:show_next_steps
echo ==========================================
echo 部署完成！后续步骤：
echo ==========================================
echo.
echo 1. 登录开发机：
echo    ssh %USERNAME%@%SERVER_IP%
echo.
echo 2. 进入项目目录：
echo    cd ~/rlhf-qwen3-baseline
echo.
echo 3. 激活环境：
echo    conda activate rlhf
echo.
echo 4. 小规模测试（推荐）：
echo    python train_with_config.py --config config_local_test.yaml
echo.
echo 5. 启动完整训练：
echo    tmux new -s training
echo    bash run_multi_gpu.sh
echo.
echo 6. 监控训练：
echo    - 查看 GPU: nvidia-smi
echo    - 查看日志: tail -f training.log
echo.
echo ==========================================

pause
