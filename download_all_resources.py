"""
一键下载所有离线资源
包括模型、数据集和Python依赖包
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"✓ {description} 完成")
        return True
    else:
        print(f"✗ {description} 失败")
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("RLHF 离线资源一键下载")
    print("="*60)
    print("将下载以下资源:")
    print("1. 模型: Qwen/Qwen3-8B (~15GB)")
    print("2. 数据集: openbmb/UltraFeedback (~800MB)")
    print("3. Python 依赖包 (~5GB)")
    print("="*60)
    
    # 确认
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 设置环境变量使用镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    success_count = 0
    total_count = 3
    
    # 1. 下载模型和数据集
    if run_command(
        "python download_assets.py",
        "下载模型和数据集"
    ):
        success_count += 1
    
    # 2. 下载依赖包 (使用清华镜像)
    if run_command(
        "pip download -r requirements.txt -d ./offline_assets/wheels -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "下载 Python 依赖包"
    ):
        success_count += 1
    
    # 3. 验证下载
    if run_command(
        "python test_offline_assets.py",
        "验证离线资源"
    ):
        success_count += 1
    
    # 总结
    print("\n" + "="*60)
    print("下载总结")
    print("="*60)
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✓ 所有资源下载完成！")
        print("\n下一步:")
        print("1. 将 offline_assets 目录传输到开发机")
        print("2. 在开发机上运行: bash install_offline.sh")
        print("3. 开始训练: python train_rlhf.py --model_name=./offline_assets/models/Qwen--Qwen3-8B")
    else:
        print("\n✗ 部分资源下载失败，请检查错误信息")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
