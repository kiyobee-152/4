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

        # NVML 相关：初始化一次，循环中只读取，避免 Windows/WDDM 下频繁 init/shutdown 导致失败
        self._nvml_inited = False
        self._nvml_handle = None
        self._gpu_available = self._check_gpu()

        # 用于避免刷屏：只在首次失败时打印 NVML 错误
        self._nvml_error_printed = False

    def _check_gpu(self) -> bool:
        """检查GPU是否可用，并缓存 NVML handle"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_inited = True
            # 默认读取 0 号 GPU；如你是多卡机器，可在此扩展为可配置
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except Exception as e:
            # 关键：不要吞异常，否则 UI 只会永远显示 0
            print(f"NVML init failed: {e!r}")
            self._nvml_inited = False
            self._nvml_handle = None
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
                if self._gpu_available and self._nvml_handle is not None:
                    try:
                        import pynvml
                        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                        gpu = float(util.gpu)
                    except Exception as e:
                        # 避免刷屏：只打印一次
                        if not self._nvml_error_printed:
                            print(f"NVML read failed: {e!r}")
                            self._nvml_error_printed = True

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

        # 关闭 NVML（如果之前 init 过）
        if self._nvml_inited:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_inited = False
            self._nvml_handle = None

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
