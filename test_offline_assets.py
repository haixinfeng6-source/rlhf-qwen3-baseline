import os
import sys
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import traceback
        from datasets import load_from_disk
"""
测试离线资源是否可用
验证模型和数据集是否正确下载
"""



def test_model(model_path):
    """测试模型加载"""
    print("=" * 60)
    print("测试模型加载")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"✗ 模型路径不存在: {model_path}")
        return False
    
    try:
        
        print("加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer 加载成功")
        print(f"  词表大小: {len(tokenizer)}")
        
        print("\n加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",  # 使用 CPU 测试，避免显存问题
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print(f"✓ 模型加载成功")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        # 测试生成
        print("\n测试生成...")
        text = "你好"
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=20)
        result = tokenizer.decode(outputs[0])
        print(f"✓ 生成测试成功")
        print(f"  输入: {text}")
        print(f"  输出: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        traceback.print_exc()
        return False


def test_dataset(dataset_path):
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("测试数据集加载")
    print("=" * 60)
    print(f"数据集路径: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"✗ 数据集路径不存在: {dataset_path}")
        return False
    
    try:
        
        print("加载数据集...")
        dataset = load_from_disk(dataset_path)
        
        print(f"✓ 数据集加载成功")
        
        # 检查数据集结构
        if hasattr(dataset, 'keys'):
            print(f"  数据集分割: {list(dataset.keys())}")
            for split_name in dataset.keys():
                print(f"  - {split_name}: {len(dataset[split_name])} 样本")
            
            # 使用训练集
            train_dataset = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        else:
            train_dataset = dataset
            print(f"  样本数: {len(train_dataset)}")
        
        # 显示第一条数据
        print(f"\n第一条数据示例:")
        print(f"  列名: {train_dataset.column_names}")
        
        first_sample = train_dataset[0]
        print(f"  Instruction: {first_sample['instruction'][:100]}...")
        if 'completions' in first_sample:
            print(f"  Completions 数量: {len(first_sample['completions'])}")
            if len(first_sample['completions']) > 0:
                print(f"  第一个 completion: {first_sample['completions'][0]['response'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        traceback.print_exc()
        return False


def test_dependencies():
    """测试依赖包"""
    print("\n" + "=" * 60)
    print("测试依赖包")
    print("=" * 60)
    
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT",
        "trl": "TRL",
        "bitsandbytes": "BitsAndBytes",
        "accelerate": "Accelerate",
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: 未安装")
            all_ok = False
    
    return all_ok


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("离线资源测试")
    print("=" * 60 + "\n")
    
    # 默认路径
    model_path = "./offline_assets/models/Qwen--Qwen3-8B"
    dataset_path = "./offline_assets/datasets/openbmb--UltraFeedback"
    
    # 如果提供了命令行参数
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        dataset_path = sys.argv[2]
    
    # 测试依赖包
    deps_ok = test_dependencies()
    
    # 测试模型
    model_ok = test_model(model_path)
    
    # 测试数据集
    dataset_ok = test_dataset(dataset_path)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"依赖包: {'✓ 通过' if deps_ok else '✗ 失败'}")
    print(f"模型: {'✓ 通过' if model_ok else '✗ 失败'}")
    print(f"数据集: {'✓ 通过' if dataset_ok else '✗ 失败'}")
    
    if deps_ok and model_ok and dataset_ok:
        print("\n✓ 所有测试通过！可以开始离线训练。")
        print("\n启动训练:")
        print(f"  python train_rlhf.py \\")
        print(f"    --model_name={model_path} \\")
        print(f"    --dataset_name={dataset_path} \\")
        print(f"    --output_dir=./output_offline")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查离线资源。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
