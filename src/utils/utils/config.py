"""
配置文件管理
"""
import yaml
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # 查找配置文件
            possible_paths = [
                "config/settings.yaml",
                "config/settings.yml",
                "../config/settings.yaml"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError("未找到配置文件")
        
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """加载配置文件"""
        logger.info(f"加载配置文件: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 设置默认值
        default_config = {
            'project': {
                'name': '城市功能区分析平台',
                'version': '1.0.0',
                'author': '未知'
            },
            'paths': {
                'data_raw': 'data/raw',
                'data_processed': 'data/processed',
                'outputs': 'outputs',
                'logs': 'logs',
                'models': 'models'
            },
            'region': {
                'name': '默认区域',
                'bounds': {
                    'min_lon': 113.3,
                    'max_lon': 113.4,
                    'min_lat': 23.1,
                    'max_lat': 23.2
                }
            }
        }
        
        # 合并配置
        config = self._merge_dicts(default_config, config)
        
        logger.info(f"配置文件加载完成: {config['project']['name']} v{config['project']['version']}")
        return config
    
    def _merge_dicts(self, dict1, dict2):
        """递归合并字典"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        # 保存到文件
        self.save_config()
    
    def save_config(self, path=None):
        """保存配置文件"""
        if path is None:
            path = self.config_path
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"配置文件已保存: {path}")
    
    def create_project_structure(self):
        """创建项目目录结构"""
        paths = self.get('paths', {})
        
        directories = [
            paths.get('data_raw', 'data/raw'),
            paths.get('data_processed', 'data/processed'),
            paths.get('outputs', 'outputs'),
            paths.get('logs', 'logs'),
            paths.get('models', 'models'),
            'config',
            'src',
            'notebooks',
            'tests',
            'docs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
        
        # 创建必要的文件
        self._create_necessary_files()
        
        logger.info("项目目录结构创建完成")
    
    def _create_necessary_files(self):
        """创建必要的文件"""
        # 创建README.md
        readme_content = f"""# {self.get('project.name')}

城市功能区动态分析平台

## 项目描述
{self.get('project.description', '基于多源数据融合的城市功能区动态分析与可视化平台')}

## 功能特性
1. 多源数据融合（POI、遥感、路网等）
2. 功能区智能识别与分类
3. 动态变化监测与分析
4. 交互式可视化展示
5. 趋势预测与发展建议

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 运行分析
python src/main.py

# 启动仪表板
python src/visualization/dashboard.py