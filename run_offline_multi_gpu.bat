@echo off
REM 离线多卡训练启动脚本 (Windows)

echo ==========================================
echo RLHF 离线多卡训练
echo ==========================================

REM 检查资源是否存在
if not exist "offline_assets\models\Qwen--Qwen2.5-7B-Instruct" (
    echo [错误] 模型文件不存在
    echo 请确保 offline_assets 目录已正确传输
    pause
    exit /b 1
)

if not exist "offline_assets\datasets\openbmb--UltraFeedback" (
    echo [错误] 数据集文件不存在
    echo 请确保 offline_assets 目录已正确传输
    pause
    exit /b 1
)

echo [检查] 资源文件完整

REM 设置训练参数
set MODEL_PATH=./offline_assets/models/Qwen--Qwen2.5-7B-Instruct
set DATASET_PATH=./offline_assets/datasets/openbmb--UltraFeedback
set OUTPUT_DIR=./output_offline

REM GPU 配置
set NUM_GPUS=4
set MASTER_PORT=29500

echo.
echo 训练配置:
echo   模型: %MODEL_PATH%
echo   数据集: %DATASET_PATH%
echo   输出: %OUTPUT_DIR%
echo   GPU 数量: %NUM_GPUS%
echo.

REM 检查 GPU
echo 检查 GPU 状态:
python -c "import torch; print(f'  可用 GPU: {torch.cuda.device_count()}')"
echo.

REM 询问是否继续
set /p CONTINUE="是否开始训练? (y/n): "
if /i not "%CONTINUE%"=="y" (
    echo 已取消
    pause
    exit /b 0
)

echo.
echo ==========================================
echo 开始多卡训练...
echo ==========================================

REM 设置可见 GPU
set CUDA_VISIBLE_DEVICES=0,1,2,3

REM 使用 torchrun 启动多卡训练
torchrun ^
    --nproc_per_node=%NUM_GPUS% ^
    --master_port=%MASTER_PORT% ^
    train_rlhf.py ^
    --model_name=%MODEL_PATH% ^
    --dataset_name=%DATASET_PATH% ^
    --output_dir=%OUTPUT_DIR% ^
    --num_train_epochs=1 ^
    --per_device_train_batch_size=1 ^
    --gradient_accumulation_steps=8 ^
    --learning_rate=1e-5 ^
    --use_lora ^
    --lora_r=16 ^
    --lora_alpha=32 ^
    --use_8bit ^
    --gradient_checkpointing ^
    --logging_steps=10 ^
    --save_steps=500 ^
    --max_length=512

echo.
echo ==========================================
echo 训练完成！
echo ==========================================
echo 模型保存在: %OUTPUT_DIR%
echo.

pause
