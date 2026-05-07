"""
无网络集群环境测试脚本
专门为离线环境设计，不检查网络连接
"""

import sys
import os
import subprocess


def check_python_version():
    """检查 Python 版本"""
    print("=" * 50)
    print("检查 Python 版本")
    print("=" * 50)
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 需要 Python 3.8+")
        return False
    print("✓ Python 版本 OK")
    return True


def check_cuda():
    """检查 CUDA 和 GPU (离线模式下允许 CPU)"""
    print("\n" + "=" * 50)
    print("检查 CUDA 和 GPU")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  显存: {props.total_memory / 1024**3:.2f} GB")
                print(f"  计算能力: {props.major}.{props.minor}")
            
            print("✓ CUDA OK")
            return True
        else:
            print("⚠ CUDA 不可用 (离线模式下允许)")
            print("  注意: 训练将使用 CPU，速度较慢")
            print("  建议在集群上安装 GPU 版 PyTorch")
            return True  # 离线模式下允许 CPU
    except ImportError:
        print("❌ PyTorch 未安装")
        return False


def check_offline_packages():
    """检查离线环境下的必要包"""
    print("\n" + "=" * 50)
    print("检查依赖包 (离线模式)")
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
            print(f"❌ {name}: 未安装")
            all_ok = False
    
    if not all_ok:
        print("\n⚠ 安装依赖包:")
        print("  bash install_offline.sh")
        print("  或")
        print("  pip install --no-index --find-links=./offline_assets/wheels -r requirements.txt")
    
    return all_ok


def check_offline_resources():
    """检查离线资源"""
    print("\n" + "=" * 50)
    print("检查离线资源")
    print("=" * 50)
    
    resources = {
        "模型文件": "./offline_assets/models/Qwen--Qwen2.5-7B-Instruct",
        "数据集": "./offline_assets/datasets/openbmb--UltraFeedback",
        "依赖包": "./offline_assets/wheels",
    }
    
    all_ok = True
    for name, path in resources.items():
        if os.path.exists(path):
            # 计算大小
            if os.path.isdir(path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                size_gb = total_size / 1024**3
                print(f"✓ {name}: {path} ({size_gb:.2f} GB)")
            else:
                print(f"✓ {name}: {path}")
        else:
            print(f"❌ {name}: 路径不存在 {path}")
            all_ok = False
    
    if not all_ok:
        print("\n⚠ 缺少离线资源:")
        print("  1. 确保已传输 offline_assets 目录到集群")
        print("  2. 检查路径是否正确")
        print("  3. 参考 离线运行指南.md 进行设置")
    
    return all_ok


def check_disk_space_offline():
    """检查磁盘空间 (离线环境降低要求)"""
    print("\n" + "=" * 50)
    print("检查磁盘空间 (离线模式)")
    print("=" * 50)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        print(f"总空间: {total / 1024**3:.2f} GB")
        print(f"已用: {used / 1024**3:.2f} GB")
        print(f"可用: {free / 1024**3:.2f} GB")
        
        # 离线环境降低要求：至少 20GB 空闲空间
        if free < 20 * 1024**3:
            print("⚠ 警告: 可用空间小于 20GB")
            print("  建议: 20GB+ 用于模型和检查点")
            return False
        
        print("✓ 磁盘空间 OK")
        return True
    except Exception as e:
        print(f"❌ 检查磁盘空间失败: {e}")
        return False


def test_offline_training():
    """测试离线训练流程"""
    print("\n" + "=" * 50)
    print("测试离线训练流程")
    print("=" * 50)
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import load_from_disk
        
        # 检查模型加载
        model_path = "./offline_assets/models/Qwen--Qwen2.5-7B-Instruct"
        if not os.path.exists(model_path):
            print(f"❌ 模型路径不存在: {model_path}")
            return False
        
        print("加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✓ Tokenizer 加载成功")
        
        # 检查数据集加载
        dataset_path = "./offline_assets/datasets/openbmb--UltraFeedback"
        if not os.path.exists(dataset_path):
            print(f"❌ 数据集路径不存在: {dataset_path}")
            return False
        
        print("加载数据集...")
        dataset = load_from_disk(dataset_path)
        print(f"✓ 数据集加载成功")
        
        if isinstance(dataset, dict):
            train_dataset = dataset.get('train', dataset[list(dataset.keys())[0]])
        else:
            train_dataset = dataset
        
        print(f"  样本数: {len(train_dataset)}")
        
        print("✓ 离线训练流程测试 OK")
        return True
        
    except Exception as e:
        print(f"❌ 离线训练流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有检查"""
    print("\n" + "=" * 60)
    print(" 无网络集群环境测试")
    print("=" * 60 + "\n")
    
    print("⚠ 注意: 此测试专为无网络集群设计")
    print("   不检查网络连接，只验证本地资源\n")
    
    results = {
        "Python 版本": check_python_version(),
        "CUDA 和 GPU": check_cuda(),
        "依赖包 (离线)": check_offline_packages(),
        "离线资源": check_offline_resources(),
        "磁盘空间 (离线)": check_disk_space_offline(),
        "离线训练流程": test_offline_training(),
    }
    
    # 总结
    print("\n" + "=" * 60)
    print(" 测试总结")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"{check:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有检查通过！可以开始离线训练。")
        print("\n启动训练:")
        print("  bash run_offline.sh              # 单卡")
        print("  bash run_offline_multi_gpu.sh    # 多卡")
    else:
        print("⚠ 部分检查失败，请按以下步骤处理:")
        print("\n1. 安装依赖包:")
        print("   bash install_offline.sh")
        print("\n2. 检查资源文件:")
        print("   ls -lh offline_assets/")
        print("\n3. 参考文档:")
        print("   离线运行指南.md")
        print("   资源清单.md")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
