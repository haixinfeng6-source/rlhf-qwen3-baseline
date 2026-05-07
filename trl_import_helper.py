#!/usr/bin/env python3
"""
TRL 导入助手
自动处理不同TRL版本的导入问题
"""

import importlib.metadata
import sys

class TRLImporter:
    """TRL导入器，自动处理版本兼容性"""
    
    def __init__(self):
        self.trl_version = None
        self.PPOConfig = None
        self.PPOTrainer = None
        self.AutoModelForCausalLMWithValueHead = None
        self._load_version()
    
    def _load_version(self):
        """加载TRL版本"""
        try:
            self.trl_version = importlib.metadata.version("trl")
            print(f"检测到 TRL 版本: {self.trl_version}")
        except importlib.metadata.PackageNotFoundError:
            print("❌ TRL 未安装")
            print("安装命令: pip install trl==0.11.0")
            sys.exit(1)
    
    def import_all(self):
        """导入所有必要的TRL组件"""
        print("\n导入TRL组件...")
        
        # 导入PPOConfig和PPOTrainer
        self._import_ppo_components()
        
        # 导入AutoModelForCausalLMWithValueHead
        self._import_value_head()
        
        # 检查导入结果
        self._check_imports()
        
        return self
    
    def _import_ppo_components(self):
        """导入PPOConfig和PPOTrainer"""
        import_attempts = []
        
        if self.trl_version.startswith('1.'):
            # TRL 1.x 版本
            import_attempts = [
                ("trl.trainer", "PPOConfig", "PPOTrainer"),
                ("trl", "PPOConfig", "PPOTrainer"),
            ]
        elif self.trl_version.startswith('0.11.'):
            # TRL 0.11.x 版本
            import_attempts = [
                ("trl", "PPOConfig", "PPOTrainer"),
                ("trl.trainer", "PPOConfig", "PPOTrainer"),
            ]
        elif self.trl_version.startswith('0.10.'):
            # TRL 0.10.x 版本
            import_attempts = [
                ("trl", "PPOConfig", "PPOTrainer"),
            ]
        else:
            # 其他版本，尝试所有可能
            import_attempts = [
                ("trl", "PPOConfig", "PPOTrainer"),
                ("trl.trainer", "PPOConfig", "PPOTrainer"),
                ("trl.core", "PPOConfig", "PPOTrainer"),
            ]
        
        for module_path, config_name, trainer_name in import_attempts:
            try:
                module = __import__(module_path, fromlist=[config_name, trainer_name])
                self.PPOConfig = getattr(module, config_name)
                self.PPOTrainer = getattr(module, trainer_name)
                print(f"✅ 从 {module_path} 导入 PPOConfig, PPOTrainer")
                return
            except (ImportError, AttributeError) as e:
                print(f"  ✗ {module_path} 导入失败: {e}")
                continue
        
        print("⚠️ 无法导入 PPOConfig 和 PPOTrainer")
    
    def _import_value_head(self):
        """导入AutoModelForCausalLMWithValueHead"""
        import_attempts = []
        
        if self.trl_version.startswith('1.'):
            # TRL 1.x 版本
            import_attempts = [
                "trl.models",
                "trl",
                "trl.models.modeling_value_head",
            ]
        elif self.trl_version.startswith('0.11.'):
            # TRL 0.11.x 版本
            import_attempts = [
                "trl.models",
                "trl",
                "trl.models.modeling_value_head",
            ]
        else:
            # 其他版本
            import_attempts = [
                "trl.models",
                "trl",
                "trl.core",
            ]
        
        for module_path in import_attempts:
            try:
                module = __import__(module_path, fromlist=["AutoModelForCausalLMWithValueHead"])
                self.AutoModelForCausalLMWithValueHead = getattr(module, "AutoModelForCausalLMWithValueHead")
                print(f"✅ 从 {module_path} 导入 AutoModelForCausalLMWithValueHead")
                return
            except (ImportError, AttributeError) as e:
                print(f"  ✗ {module_path} 导入失败: {e}")
                continue
        
        print("⚠️ 无法导入 AutoModelForCausalLMWithValueHead")
    
    def _check_imports(self):
        """检查导入结果"""
        print("\n导入结果检查:")
        
        if self.PPOConfig:
            print(f"  PPOConfig: ✅ ({self.PPOConfig.__module__})")
        else:
            print("  PPOConfig: ❌ 未导入")
        
        if self.PPOTrainer:
            print(f"  PPOTrainer: ✅ ({self.PPOTrainer.__module__})")
        else:
            print("  PPOTrainer: ❌ 未导入")
        
        if self.AutoModelForCausalLMWithValueHead:
            print(f"  AutoModelForCausalLMWithValueHead: ✅ ({self.AutoModelForCausalLMWithValueHead.__module__})")
        else:
            print("  AutoModelForCausalLMWithValueHead: ❌ 未导入")
    
    def get_import_statements(self):
        """获取正确的导入语句"""
        if not all([self.PPOConfig, self.PPOTrainer, self.AutoModelForCausalLMWithValueHead]):
            return None
        
        config_module = self.PPOConfig.__module__
        trainer_module = self.PPOTrainer.__module__
        valuehead_module = self.AutoModelForCausalLMWithValueHead.__module__
        
        imports = []
        
        # PPOConfig和PPOTrainer
        if config_module == trainer_module:
            imports.append(f"from {config_module} import PPOConfig, PPOTrainer")
        else:
            imports.append(f"from {config_module} import PPOConfig")
            imports.append(f"from {trainer_module} import PPOTrainer")
        
        # AutoModelForCausalLMWithValueHead
        imports.append(f"from {valuehead_module} import AutoModelForCausalLMWithValueHead")
        
        return "\n".join(imports)

def test_import():
    """测试导入"""
    print("=" * 60)
    print("TRL 导入测试")
    print("=" * 60)
    
    importer = TRLImporter()
    importer.import_all()
    
    print("\n" + "=" * 60)
    print("推荐导入语句:")
    print("=" * 60)
    
    import_statements = importer.get_import_statements()
    if import_statements:
        print(import_statements)
        print("\n使用示例:")
        print("```python")
        print("# 在代码开头添加:")
        print("from trl_import_helper import TRLImporter")
        print("trl = TRLImporter().import_all()")
        print("PPOConfig = trl.PPOConfig")
        print("PPOTrainer = trl.PPOTrainer")
        print("AutoModelForCausalLMWithValueHead = trl.AutoModelForCausalLMWithValueHead")
        print("```")
    else:
        print("无法确定正确的导入语句")
        print("\n建议:")
        print("1. 安装特定版本: pip install trl==0.11.0")
        print("2. 检查TRL结构: python check_trl_structure.py")
        print("3. 使用动态导入")
    
    return importer

if __name__ == "__main__":
    test_import()