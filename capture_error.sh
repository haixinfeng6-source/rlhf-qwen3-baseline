#!/bin/bash

# 捕获完整错误信息的脚本

echo "=========================================="
echo "捕获完整错误信息"
echo "=========================================="

# 设置环境变量以获取更多调试信息
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

# 临时输出文件
ERROR_LOG="torchrun_error_$(date +%Y%m%d_%H%M%S).log"

echo "错误日志将保存到: $ERROR_LOG"
echo "开始运行..."

# 使用 tee 同时输出到屏幕和文件
torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    train_rlhf.py \
    --model_name=Qwen/Qwen3-8B \
    --dataset_name=openbmb/UltraFeedback \
    --output_dir="./test_output" \
    --max_samples=5 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=1 \
    --use_lora \
    --use_8bit \
    --gradient_checkpointing \
    --logging_steps=1 \
    2>&1 | tee "$ERROR_LOG"

echo "=========================================="
echo "运行完成"
echo "错误日志已保存到: $ERROR_LOG"
echo "=========================================="

# 显示错误日志的最后50行
echo "错误日志最后50行:"
tail -50 "$ERROR_LOG"