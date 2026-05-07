#!/usr/bin/env python3
"""
单进程运行脚本，避免 torchrun 复杂性
"""

import os
import sys

# 设置单进程环境变量
os.environ['LOCAL_RANK'] = '-1'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

print("=" * 60)
print("单进程测试 (不使用 torchrun)")
print("=" * 60)

# 直接运行 train_rlhf.py 的 main 函数
if __name__ == "__main__":
    try:
        # 导入 train_rlhf 模块
        import train_rlhf
        
        print("✓ train_rlhf 模块导入成功")
        print("开始运行 main() 函数...")
        
        # 运行 main 函数
        train_rlhf.main()
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("运行失败!")
        print("=" * 60)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print("\n完整堆栈跟踪:")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("建议:")
        print("1. 检查 Python 版本 (推荐 3.10)")
        print("2. 检查 TRL 版本: pip show trl")
        print("3. 检查依赖: pip install -r requirements.txt")
        print("=" * 60)
        
        sys.exit(1)