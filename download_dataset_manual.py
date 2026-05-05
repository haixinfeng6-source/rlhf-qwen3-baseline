"""
手动下载 UltraFeedback 数据集
提供多种下载方式和详细的错误处理
"""

import os
import sys
from datasets import load_dataset
from huggingface_hub import snapshot_download


def method_1_load_dataset():
    """
    方法 1: 使用 datasets.load_dataset
    """
    print("=" * 60)
    print("方法 1: 使用 datasets.load_dataset")
    print("=" * 60)
    
    try:
        print("正在下载 UltraFeedback 数据集...")
        print("这可能需要几分钟，请耐心等待...")
        
        # 下载数据集
        dataset = load_dataset("openbmb/UltraFeedback")
        
        # 保存到本地
        save_path = "./offline_assets/datasets/openbmb--UltraFeedback"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"保存数据集到: {save_path}")
        dataset.save_to_disk(save_path)
        
        print(f"✓ 数据集下载成功！")
        print(f"  训练集: {len(dataset['train'])} 样本")
        
        # 显示第一条数据
        print(f"\n第一条数据示例:")
        first = dataset['train'][0]
        print(f"  Instruction: {first['instruction'][:100]}...")
        print(f"  Completions: {len(first['completions'])} 个")
        
        return True
        
    except Exception as e:
        print(f"✗ 方法 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def method_2_snapshot_download():
    """
    方法 2: 使用 snapshot_download 下载整个仓库
    """
    print("\n" + "=" * 60)
    print("方法 2: 使用 snapshot_download")
    print("=" * 60)
    
    try:
        print("正在下载 UltraFeedback 数据集仓库...")
        
        save_path = "./offline_assets/datasets/openbmb--UltraFeedback-raw"
        
        snapshot_download(
            repo_id="openbmb/UltraFeedback",
            repo_type="dataset",
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"✓ 数据集仓库下载成功: {save_path}")
        print(f"  请手动加载数据集")
        
        return True
        
    except Exception as e:
        print(f"✗ 方法 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def method_3_streaming():
    """
    方法 3: 使用流式加载并保存部分数据
    """
    print("\n" + "=" * 60)
    print("方法 3: 流式加载并保存")
    print("=" * 60)
    
    try:
        print("正在流式加载数据集...")
        
        # 流式加载
        dataset = load_dataset("openbmb/UltraFeedback", streaming=True)
        
        print("✓ 流式加载成功")
        print("  注意: 流式模式不会下载完整数据集")
        print("  建议使用方法 1 或 2 下载完整数据集")
        
        # 显示前几条数据
        print("\n前 3 条数据:")
        for i, sample in enumerate(dataset['train'].take(3)):
            print(f"\n样本 {i+1}:")
            print(f"  Instruction: {sample['instruction'][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 方法 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("检查环境配置")
    print("=" * 60)
    
    # 检查 datasets 版本
    try:
        import datasets
        print(f"✓ datasets 版本: {datasets.__version__}")
    except ImportError:
        print("✗ datasets 未安装")
        print("  安装: pip install datasets")
        return False
    
    # 检查 huggingface_hub 版本
    try:
        import huggingface_hub
        print(f"✓ huggingface_hub 版本: {huggingface_hub.__version__}")
    except ImportError:
        print("✗ huggingface_hub 未安装")
        print("  安装: pip install huggingface_hub")
        return False
    
    # 检查网络连接
    print("\n检查网络连接...")
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        print("✓ 可以访问 HuggingFace")
    except:
        print("✗ 无法访问 HuggingFace")
        print("  解决方案:")
        print("  1. 检查网络连接")
        print("  2. 使用镜像: export HF_ENDPOINT=https://hf-mirror.com")
        print("  3. 使用代理")
        return False
    
    # 检查磁盘空间
    print("\n检查磁盘空间...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"✓ 可用空间: {free_gb:.2f} GB")
        
        if free_gb < 5:
            print("⚠ 警告: 可用空间不足 5GB")
            print("  UltraFeedback 数据集约需 2GB")
            return False
    except:
        print("⚠ 无法检查磁盘空间")
    
    return True


def download_with_mirror():
    """使用国内镜像下载"""
    print("\n" + "=" * 60)
    print("使用国内镜像下载")
    print("=" * 60)
    
    # 设置镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("已设置 HF_ENDPOINT=https://hf-mirror.com")
    
    return method_1_load_dataset()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("UltraFeedback 数据集下载工具")
    print("=" * 60 + "\n")
    
    # 检查环境
    if not check_environment():
        print("\n环境检查失败，请先解决上述问题")
        return 1
    
    print("\n" + "=" * 60)
    print("开始下载数据集")
    print("=" * 60)
    
    # 尝试方法 1
    if method_1_load_dataset():
        print("\n✓ 下载成功！")
        return 0
    
    # 如果方法 1 失败，尝试使用镜像
    print("\n尝试使用国内镜像...")
    if download_with_mirror():
        print("\n✓ 使用镜像下载成功！")
        return 0
    
    # 如果还是失败，尝试方法 2
    print("\n尝试方法 2...")
    if method_2_snapshot_download():
        print("\n✓ 方法 2 成功！")
        print("  数据集已下载，但需要手动加载")
        return 0
    
    # 最后尝试流式加载
    print("\n尝试方法 3（流式加载）...")
    if method_3_streaming():
        print("\n⚠ 流式加载成功，但未下载完整数据集")
        print("  建议:")
        print("  1. 检查网络连接")
        print("  2. 使用代理或镜像")
        print("  3. 手动从 HuggingFace 网站下载")
        return 1
    
    # 所有方法都失败
    print("\n" + "=" * 60)
    print("所有下载方法都失败")
    print("=" * 60)
    print("\n可能的原因:")
    print("1. 网络问题 - 无法访问 HuggingFace")
    print("2. 数据集需要认证 - 需要登录 HuggingFace")
    print("3. 磁盘空间不足")
    print("\n解决方案:")
    print("1. 使用镜像:")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    print("   python download_dataset_manual.py")
    print("\n2. 登录 HuggingFace:")
    print("   huggingface-cli login")
    print("   python download_dataset_manual.py")
    print("\n3. 手动下载:")
    print("   访问: https://huggingface.co/datasets/openbmb/UltraFeedback")
    print("   下载后放到: ./offline_assets/datasets/openbmb--UltraFeedback")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
