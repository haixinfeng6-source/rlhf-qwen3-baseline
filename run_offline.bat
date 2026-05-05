@echo off
REM 离线训练启动脚本 (Windows)

echo ==========================================
echo RLHF 离线训练
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

echo.
echo 训练配置:
echo   模型: %MODEL_PATH%
echo   数据集: %DATASET_PATH%
echo   输出: %OUTPUT_DIR%
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
echo 开始训练...
echo ==========================================

python train_rlhf.py ^
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
