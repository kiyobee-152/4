# -*- coding: utf-8 -*-
"""
系统性能监测模块
监测 CPU、内存、GPU 占用率
"""
import psutil
import threading
import time
from typing import Dict


class SystemMonitor:
    """系统监测器"""
    
    def __init__(self, update_interval=2):
        """
        初始化系统监测器
        
        Args:
            update_interval: 更新间隔（秒）
        """
        self.update_interval = update_interval
        self.metrics = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_percent': 0.0
        }
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._gpu_available = self._check_gpu()
        
    def _check_gpu(self) -> bool:
        """检查GPU是否可用"""
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except:
            return False
    
    def _monitor_loop(self):
        """监测循环"""
        while self._running:
            try:
                # CPU占用率
                cpu = psutil.cpu_percent(interval=0.1)
                
                # 内存占用率
                memory = psutil.virtual_memory().percent
                
                # GPU占用率（如果可用）
                gpu = 0.0
                if self._gpu_available:
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu = util.gpu
                        pynvml.nvmlShutdown()
                    except:
                        pass
                
                # 更新指标
                with self._lock:
                    self.metrics['cpu_percent'] = cpu
                    self.metrics['memory_percent'] = memory
                    self.metrics['gpu_percent'] = gpu
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"监测错误: {e}")
                time.sleep(self.update_interval)
    
    def start(self):
        """启动监测线程"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
    
    def stop(self):
        """停止监测线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def get_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        with self._lock:
            return self.metrics.copy()


# 全局监测器实例
_monitor = SystemMonitor()
_monitor.start()


def get_system_metrics() -> Dict[str, float]:
    """获取系统监测指标"""
    return _monitor.get_metrics()