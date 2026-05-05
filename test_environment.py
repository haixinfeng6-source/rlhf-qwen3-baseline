"""
环境测试脚本 - 在训练前验证环境配置
"""

import sys
import subprocess


def check_python_version():
    """检查 Python 版本"""
    print("=" * 50)
    print("Checking Python Version")
    print("=" * 50)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    return True


def check_cuda():
    """检查 CUDA 和 GPU"""
    print("\n" + "=" * 50)
    print("Checking CUDA and GPU")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
            
            print("✓ CUDA OK")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_packages():
    """检查必要的包"""
    print("\n" + "=" * 50)
    print("Checking Required Packages")
    print("=" * 50)
    
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT",
        "trl": "TRL",
        "bitsandbytes": "BitsAndBytes",
        "accelerate": "Accelerate",
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: Not installed")
            all_ok = False
    
    return all_ok


def check_disk_space():
    """检查磁盘空间"""
    print("\n" + "=" * 50)
    print("Checking Disk Space")
    print("=" * 50)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        print(f"Total: {total / 1024**3:.2f} GB")
        print(f"Used: {used / 1024**3:.2f} GB")
        print(f"Free: {free / 1024**3:.2f} GB")
        
        # 建议至少 50GB 空闲空间
        if free < 50 * 1024**3:
            print("⚠ Warning: Less than 50GB free space")
            print("  Recommended: 50GB+ for model and checkpoints")
            return False
        
        print("✓ Disk space OK")
        return True
    except Exception as e:
        print(f"❌ Error checking disk space: {e}")
        return False


def check_network():
    """检查网络连接（HuggingFace）"""
    print("\n" + "=" * 50)
    print("Checking Network Connection")
    print("=" * 50)
    
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        print("✓ HuggingFace accessible")
        return True
    except:
        print("⚠ Warning: Cannot access HuggingFace")
        print("  You may need to set HF_ENDPOINT or use proxy")
        return False


def test_simple_training():
    """测试简单的训练流程"""
    print("\n" + "=" * 50)
    print("Testing Simple Training Flow")
    print("=" * 50)
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Testing generation...")
        inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=20)
        result = tokenizer.decode(outputs[0])
        
        print(f"Generated: {result}")
        print("✓ Training flow test OK")
        return True
        
    except Exception as e:
        print(f"❌ Training flow test failed: {e}")
        return False


def main():
    """运行所有检查"""
    print("\n" + "=" * 60)
    print(" RLHF Training Environment Test")
    print("=" * 60 + "\n")
    
    results = {
        "Python Version": check_python_version(),
        "CUDA & GPU": check_cuda(),
        "Required Packages": check_packages(),
        "Disk Space": check_disk_space(),
        "Network": check_network(),
        "Training Flow": test_simple_training(),
    }
    
    # 总结
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! Ready to start training.")
    else:
        print("⚠ Some checks failed. Please fix the issues before training.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
