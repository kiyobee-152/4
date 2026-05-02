# -*- coding: utf-8 -*-
"""
配置管理模块
保存和恢复用户参数设置
"""
import json
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        'confidence': 0.45,
        'iou': 0.45,
        'weights': '',
        'device': 'gpu',
        'enhancement': False
    }
    
    def __init__(self, config_file='./config.json'):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置和加载的配置
                    return {**self.DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"配置加载失败: {e}")
        
        # 返回默认配置
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """保存配置文件"""
        try:
            if config:
                self.config = config
            
            os.makedirs(os.path.dirname(self.config_file) or '.', exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"配置保存失败: {e}")
    
    def get(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value
        self.save_config()
    
    def update(self, updates: Dict[str, Any]):
        """更新多个配置值"""
        self.config.update(updates)
        self.save_config()


# 全局配置管理器实例
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    return _config_manager