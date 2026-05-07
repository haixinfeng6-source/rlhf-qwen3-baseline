@echo off
REM RLHF 多卡训练启动脚本 (Windows)
REM 使用标准 PPO 方法
REM 支持 4卡 和 8卡 训练

echo ==========================================
echo RLHF (PPO) Multi-GPU Training (Windows)
echo ==========================================

REM 设置环境变量
set CUDA_VISIBLE_DEVICES=0,1,2,3
set NCCL_DEBUG=INFO

REM GPU 数量
set NUM_GPUS=4

REM 模型和数据集配置
set MODEL_NAME=Qwen/Qwen3-8B
set DATASET_NAME=openbmb/UltraFeedback
set OUTPUT_DIR=./output_rlhf_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%

REM 训练超参数
set BATCH_SIZE=1
set GRAD_ACCUM=8
set LEARNING_RATE=1e-5
set EPOCHS=1
set MAX_LENGTH=512

REM LoRA 配置
set LORA_R=16
set LORA_ALPHA=32

echo ==========================================
echo Starting RLHF (PPO) training with %NUM_GPUS% GPUs
echo Model: %MODEL_NAME%
echo Dataset: %DATASET_NAME%
echo Output: %OUTPUT_DIR%
echo ==========================================

REM 使用 torchrun 启动多卡训练
torchrun ^
    --nproc_per_node=%NUM_GPUS% ^
    --master_port=29500 ^
    train_rlhf.py ^
    --model_name=%MODEL_NAME% ^
    --dataset_name=%DATASET_NAME% ^
    --output_dir=%OUTPUT_DIR% ^
    --per_device_train_batch_size=%BATCH_SIZE% ^
    --gradient_accumulation_steps=%GRAD_ACCUM% ^
    --learning_rate=%LEARNING_RATE% ^
    --num_train_epochs=%EPOCHS% ^
    --max_length=%MAX_LENGTH% ^
    --use_lora ^
    --lora_r=%LORA_R% ^
    --lora_alpha=%LORA_ALPHA% ^
    --use_8bit ^
    --gradient_checkpointing ^
    --logging_steps=10 ^
    --save_steps=500

echo ==========================================
echo Training completed!
echo Model saved to: %OUTPUT_DIR%
echo ==========================================

pause
