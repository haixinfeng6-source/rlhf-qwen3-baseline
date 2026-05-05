"""
本地测试脚本 - 用于验证代码是否可以运行
使用小数据集快速测试
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def test_environment():
    """测试环境配置"""
    print("=" * 50)
    print("环境测试")
    print("=" * 50)
    
    # 检查 CUDA
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ 警告: 没有检测到 GPU，训练会非常慢")
    
    print()

def test_model_loading():
    """测试模型加载"""
    print("=" * 50)
    print("测试模型加载（使用小模型）")
    print("=" * 50)
    
    try:
        # 使用小模型测试
        model_name = "gpt2"  # 小模型，快速测试
        print(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print("✓ 模型加载成功")
        
        # 测试生成
        text = "Hello, how are you?"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=30)
        result = tokenizer.decode(outputs[0])
        
        print(f"测试生成: {result}")
        print("✓ 模型生成成功")
        
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "=" * 50)
    print("测试数据集加载")
    print("=" * 50)
    
    try:
        print("加载 UltraFeedback 数据集（前 10 条）...")
        dataset = load_dataset("openbmb/UltraFeedback", split="train[:10]")
        print(f"✓ 数据集加载成功，样本数: {len(dataset)}")
        print(f"数据集列: {dataset.column_names}")
        
        # 显示第一条数据
        print("\n第一条数据示例:")
        print(f"Instruction: {dataset[0]['instruction'][:100]}...")
        print(f"Completions 数量: {len(dataset[0]['completions'])}")
        
        return True
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("提示: 可能需要网络连接或设置 HF_ENDPOINT")
        return False

def estimate_memory_requirements():
    """估算显存需求"""
    print("\n" + "=" * 50)
    print("显存需求估算（Qwen3-8B）")
    print("=" * 50)
    
    configs = [
        ("FP16 全参数", 16),
        ("FP16 + LoRA(r=16)", 12),
        ("8bit + LoRA(r=16)", 8),
        ("4bit + LoRA(r=16)", 6),
        ("4bit + LoRA(r=8)", 5),
    ]
    
    print("\n单卡显存需求:")
    for config, memory in configs:
        print(f"  {config:.<30} {memory} GB")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n你的 GPU 显存: {gpu_memory:.2f} GB")
        
        if gpu_memory >= 24:
            print("✓ 显存充足，可以使用 8bit + LoRA 配置")
        elif gpu_memory >= 16:
            print("⚠️ 显存有限，建议使用 4bit + LoRA 配置")
        else:
            print("❌ 显存不足，可能无法训练 Qwen3-8B")
            print("   建议: 使用云服务器或更小的模型")

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" 本地环境测试")
    print("=" * 60 + "\n")
    
    # 运行测试
    test_environment()
    model_ok = test_model_loading()
    dataset_ok = test_dataset_loading()
    estimate_memory_requirements()
    
    # 总结
    print("\n" + "=" * 60)
    print(" 测试总结")
    print("=" * 60)
    
    if model_ok and dataset_ok:
        print("✓ 所有测试通过！")
        print("\n你可以开始训练了:")
        print("  python train_rlhf.py")
        print("\n或者使用配置文件:")
        print("  python train_with_config.py --config config.yaml")
    else:
        print("⚠️ 部分测试失败")
        print("\n请检查:")
        if not model_ok:
            print("  - 模型加载问题: 检查 transformers 版本")
        if not dataset_ok:
            print("  - 数据集加载问题: 检查网络连接或设置镜像")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
