"""
离线资源下载脚本
在有网络的机器上运行此脚本，下载模型和数据集
然后将下载的文件传输到开发机
"""

import os
import argparse
from huggingface_hub import snapshot_download
from datasets import load_dataset


def download_model(model_name: str, save_dir: str = "./models"):
    """
    下载模型到本地
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen2.5-7B-Instruct"
        save_dir: 保存目录
    """
    print(f"=" * 60)
    print(f"下载模型: {model_name}")
    print(f"保存到: {save_dir}")
    print(f"=" * 60)
    
    model_path = os.path.join(save_dir, model_name.replace("/", "--"))
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ 模型下载完成: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        return None


def download_dataset(dataset_name: str, save_dir: str = "./datasets"):
    """
    下载数据集到本地
    
    Args:
        dataset_name: 数据集名称，如 "openbmb/UltraFeedback"
        save_dir: 保存目录
    """
    print(f"\n" + "=" * 60)
    print(f"下载数据集: {dataset_name}")
    print(f"保存到: {save_dir}")
    print(f"=" * 60)
    
    dataset_path = os.path.join(save_dir, dataset_name.replace("/", "--"))
    
    try:
        # 下载数据集
        dataset = load_dataset(dataset_name)
        
        # 保存到本地
        dataset.save_to_disk(dataset_path)
        
        print(f"✓ 数据集下载完成: {dataset_path}")
        print(f"  - 训练集: {len(dataset['train'])} 样本")
        if 'test' in dataset:
            print(f"  - 测试集: {len(dataset['test'])} 样本")
        
        return dataset_path
    except Exception as e:
        print(f"✗ 数据集下载失败: {e}")
        return None


def download_dependencies(save_dir: str = "./wheels"):
    """
    下载 Python 依赖包
    
    Args:
        save_dir: 保存目录
    """
    print(f"\n" + "=" * 60)
    print(f"下载 Python 依赖包")
    print(f"保存到: {save_dir}")
    print(f"=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取 requirements.txt (使用 UTF-8 编码)
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = f.read()
    except UnicodeDecodeError:
        # 如果 UTF-8 失败，尝试其他编码
        with open("requirements.txt", "r", encoding="gbk") as f:
            requirements = f.read()
    
    print("执行命令:")
    cmd = f"pip download -r requirements.txt -d {save_dir}"
    print(f"  {cmd}")
    print("\n请手动执行上述命令下载依赖包")
    print(f"或运行: python -m pip download -r requirements.txt -d {save_dir}")


def create_offline_install_script():
    """创建离线安装脚本"""
    
    # Linux 脚本
    install_script = """#!/bin/bash

# 离线安装脚本
# 在开发机上运行此脚本安装依赖

echo "=========================================="
echo "离线安装 Python 依赖"
echo "=========================================="

# 安装依赖包
if [ -d "wheels" ]; then
    echo "从 wheels 目录安装依赖..."
    pip install --no-index --find-links=wheels -r requirements.txt
    echo "✓ 依赖安装完成"
else
    echo "✗ 未找到 wheels 目录"
    echo "请确保已将 wheels 目录上传到开发机"
    exit 1
fi

echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"

echo ""
echo "安装完成！"
"""
    
    with open("install_offline.sh", "w") as f:
        f.write(install_script)
    
    print(f"\n✓ 创建离线安装脚本: install_offline.sh")
    
    # Windows 脚本
    install_script_bat = """@echo off

REM 离线安装脚本 (Windows)
REM 在开发机上运行此脚本安装依赖

echo ==========================================
echo 离线安装 Python 依赖
echo ==========================================

if exist wheels (
    echo 从 wheels 目录安装依赖...
    pip install --no-index --find-links=wheels -r requirements.txt
    echo 依赖安装完成
) else (
    echo 未找到 wheels 目录
    echo 请确保已将 wheels 目录上传到开发机
    exit /b 1
)

echo.
echo ==========================================
echo 验证安装
echo ==========================================

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"

echo.
echo 安装完成！
pause
"""
    
    with open("install_offline.bat", "w") as f:
        f.write(install_script_bat)
    
    print(f"✓ 创建离线安装脚本: install_offline.bat")


def main():
    parser = argparse.ArgumentParser(description="下载 RLHF 训练所需的离线资源")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="模型名称"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openbmb/UltraFeedback",
        help="数据集名称"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./offline_assets/models",
        help="模型保存目录"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./offline_assets/datasets",
        help="数据集保存目录"
    )
    parser.add_argument(
        "--wheels-dir",
        type=str,
        default="./offline_assets/wheels",
        help="依赖包保存目录"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="跳过模型下载"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="跳过数据集下载"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("RLHF 离线资源下载工具")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print("=" * 60 + "\n")
    
    # 创建目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.wheels_dir, exist_ok=True)
    
    # 下载模型
    if not args.skip_model:
        model_path = download_model(args.model, args.model_dir)
    else:
        print("跳过模型下载")
        model_path = None
    
    # 下载数据集
    if not args.skip_dataset:
        dataset_path = download_dataset(args.dataset, args.dataset_dir)
    else:
        print("跳过数据集下载")
        dataset_path = None
    
    # 下载依赖包
    download_dependencies(args.wheels_dir)
    
    # 创建离线安装脚本
    create_offline_install_script()
    
    # 总结
    print("\n" + "=" * 60)
    print("下载完成！")
    print("=" * 60)
    
    if model_path:
        print(f"✓ 模型: {model_path}")
    if dataset_path:
        print(f"✓ 数据集: {dataset_path}")
    print(f"✓ 依赖包: {args.wheels_dir} (需手动下载)")
    
    print("\n下一步:")
    print("1. 下载依赖包:")
    print(f"   pip download -r requirements.txt -d {args.wheels_dir}")
    print("\n2. 将整个 offline_assets 目录传输到开发机:")
    print(f"   scp -r offline_assets username@server:/path/to/project/")
    print("\n3. 在开发机上运行:")
    print("   bash install_offline.sh")
    print("   python train_rlhf.py --model_name=./offline_assets/models/Qwen--Qwen2.5-7B-Instruct")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
