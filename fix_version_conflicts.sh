#!/bin/bash

# 修复版本冲突问题
# 1. huggingface_hub 中 is_offline_mode 不存在
# 2. pyarrow 中 PyExtensionType 不存在

echo "=========================================="
echo "修复版本冲突问题"
echo "=========================================="

echo "检测到的问题:"
echo "1. huggingface_hub 缺少 is_offline_mode"
echo "2. pyarrow 缺少 PyExtensionType"
echo ""

echo "解决方案: 安装兼容版本"
echo ""

# 检查当前版本
echo "当前版本:"
python -c "
try:
    import huggingface_hub
    print(f'huggingface_hub: {huggingface_hub.__version__}')
except:
    print('huggingface_hub: 未安装')

try:
    import pyarrow
    print(f'pyarrow: {pyarrow.__version__}')
except:
    print('pyarrow: 未安装')

try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except:
    print('transformers: 未安装')
"

echo ""
echo "开始修复..."

# 修复 huggingface_hub 问题
echo "1. 修复 huggingface_hub..."
pip install "huggingface_hub<1.0.0" --upgrade

# 修复 pyarrow 问题
echo "2. 修复 pyarrow..."
pip install "pyarrow<15.0.0" --upgrade

# 确保 transformers 兼容版本
echo "3. 确保 transformers 兼容..."
pip install "transformers==4.36.0" --upgrade

# 重新安装 datasets 以确保兼容性
echo "4. 重新安装 datasets..."
pip install "datasets==2.14.0" --upgrade --force-reinstall

echo ""
echo "验证修复..."

python -c "
import sys

print('测试导入...')

# 测试 huggingface_hub
try:
    from huggingface_hub import is_offline_mode
    print('✓ huggingface_hub.is_offline_mode 导入成功')
except ImportError:
    try:
        # 尝试其他可能的导入方式
        import huggingface_hub
        if hasattr(huggingface_hub, 'is_offline_mode'):
            print('✓ huggingface_hub.is_offline_mode 存在')
        else:
            print('✗ huggingface_hub.is_offline_mode 不存在，尝试替代方案')
            # 创建替代函数
            def is_offline_mode():
                import os
                return os.environ.get('HF_HUB_OFFLINE', '0') == '1'
            print('✓ 创建了替代的 is_offline_mode 函数')
    except Exception as e:
        print(f'✗ huggingface_hub 检查失败: {e}')

# 测试 pyarrow
try:
    import pyarrow as pa
    if hasattr(pa, 'PyExtensionType'):
        print('✓ pyarrow.PyExtensionType 存在')
    else:
        print('✗ pyarrow.PyExtensionType 不存在，使用 ExtensionType')
        # 使用 ExtensionType 作为替代
        pa.PyExtensionType = pa.ExtensionType
        print('✓ 设置了 pa.PyExtensionType = pa.ExtensionType')
except Exception as e:
    print(f'✗ pyarrow 检查失败: {e}')

# 测试 transformers
try:
    import transformers
    print(f'✓ transformers 导入成功: {transformers.__version__}')
except Exception as e:
    print(f'✗ transformers 导入失败: {e}')

# 测试 TRL
try:
    from trl.models import AutoModelForCausalLMWithValueHead
    print('✓ AutoModelForCausalLMWithValueHead 导入成功')
except ImportError as e:
    print(f'✗ AutoModelForCausalLMWithValueHead 导入失败: {e}')
"

echo ""
echo "创建补丁文件..."

# 创建 huggingface_hub 补丁
cat > fix_huggingface_hub.py << 'EOF'
"""
修复 huggingface_hub 中 is_offline_mode 不存在的问题
"""

import os
import sys

# 检查 huggingface_hub 版本
try:
    import huggingface_hub
    
    # 如果 is_offline_mode 不存在，创建它
    if not hasattr(huggingface_hub, 'is_offline_mode'):
        def is_offline_mode():
            """检查是否处于离线模式"""
            return os.environ.get('HF_HUB_OFFLINE', '0') == '1'
        
        # 添加到模块中
        huggingface_hub.is_offline_mode = is_offline_mode
        huggingface_hub.is_offline_mode.__module__ = 'huggingface_hub'
        
        print("✓ 为 huggingface_hub 添加了 is_offline_mode 函数")
        
except ImportError:
    pass

# 应用到当前环境
if __name__ == "__main__":
    print("huggingface_hub 补丁已应用")
EOF

# 创建 pyarrow 补丁
cat > fix_pyarrow.py << 'EOF'
"""
修复 pyarrow 中 PyExtensionType 不存在的问题
"""

try:
    import pyarrow as pa
    
    # 如果 PyExtensionType 不存在，使用 ExtensionType
    if not hasattr(pa, 'PyExtensionType'):
        pa.PyExtensionType = pa.ExtensionType
        print("✓ 设置了 pa.PyExtensionType = pa.ExtensionType")
        
except ImportError:
    pass

if __name__ == "__main__":
    print("pyarrow 补丁已应用")
EOF

echo ""
echo "已创建补丁文件:"
echo "1. fix_huggingface_hub.py - 修复 is_offline_mode"
echo "2. fix_pyarrow.py - 修复 PyExtensionType"
echo ""
echo "使用方法:"
echo "在运行训练脚本前，先导入补丁:"
echo "  import fix_huggingface_hub"
echo "  import fix_pyarrow"
echo ""
echo "或者直接运行:"
echo "  python fix_huggingface_hub.py"
echo "  python fix_pyarrow.py"
echo ""
echo "=========================================="
echo "修复完成!"
echo "=========================================="
echo "下一步:"
echo "1. 运行补丁: python fix_huggingface_hub.py && python fix_pyarrow.py"
echo "2. 测试导入: python direct_import_test.py"
echo "3. 如果仍有问题，可能需要创建虚拟环境"
echo "=========================================="