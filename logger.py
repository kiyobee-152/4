# -*- coding: utf-8 -*-
"""
统一日志管理模块
提供日志记录功能：文件日志 + 控制台输出
"""
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir='./logs', log_level=logging.INFO):
    """
    配置日志系统
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    log_file = os.path.join(log_dir, 'app.log')
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除已存在的handlers
    logger.handlers.clear()
    
    # 文件handler（RotatingFileHandler，每日轮转，保留7天）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=7
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # 添加handlers到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name=None):
    """获取logger实例"""
    return logging.getLogger(name)