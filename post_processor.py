# -*- coding: utf-8 -*-
"""
后处理模块
提供检测结果的后处理功能：保存结果、导出报告、统计信息等

本模块负责管理目标检测完成后的所有下游处理任务，是整个系统的"数据出口"。
包含两个核心类：
  - DetectionResult（检测结果数据类）：表示单条检测记录的数据结构，封装了类别名称、
    置信度、边界框坐标、时间戳、帧ID和镜头ID等信息，并提供了 to_dict() 方法用于序列化。
  - PostProcessor（后处理器类）：管理所有检测结果的存储、统计和导出，维护一个
    detection_history 列表作为检测历史记录，以及一个 statistics 字典进行类别计数。
    提供了多种格式的导出能力（TXT 报告、CSV 表格、JSON 数据）和图像/视频保存功能。

在系统中的调用位置（main.py）：
  - Ui_MainWindow.__init__ 中创建 PostProcessor 实例
  - start_camera() / start_video() / start_image() 中每帧检测后调用 add_detection() 记录结果
  - get_result_str() 中调用 get_statistics() 获取累计统计用于界面显示
  - GUI 按钮分别绑定 export_report()、export_csv()、export_json()、save_current_image()
  - "清空检测记录"按钮绑定 clear_history()

依赖关系：
  - os：文件路径操作、目录创建
  - json：JSON 格式导出
  - csv：CSV 格式导出
  - cv2（OpenCV）：图像和视频的读写保存
  - numpy：图像数组类型
  - datetime：时间戳生成
  - collections.defaultdict：类别计数的自动初始化字典
"""
import os
import json
import csv
import threading
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# =============================================================================
# DetectionResult - 检测结果数据类
# =============================================================================
class DetectionResult:
    """检测结果数据类"""
    
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int],
                 timestamp: Optional[str] = None, frame_id: Optional[int] = None,
                 feed_id: Optional[int] = None):
        """
        初始化检测结果
        
        Args:
            class_name: 类别名称
            confidence: 置信度
            bbox: 边界框 (x1, y1, x2, y2)
            timestamp: 时间戳
            frame_id: 帧ID
            feed_id: 镜头ID (0-7，表示 8 个画面中的哪一个)
        """
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.frame_id = frame_id
        # ✅ 新增：镜头ID，标识来自 8 个画面中的哪一个
        # 0-7 分别代表"画面1"-"画面8"
        self.feed_id = feed_id if feed_id is not None else 0
    
    def to_dict(self) -> Dict:
        """
        将检测结果转换为字典格式
        
        用于 JSON 序列化导出。将所有属性转为 Python 基本类型。
        
        Returns:
            包含所有检测信息的字典，键为英文字段名
        """
        return {
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': list(self.bbox),
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'feed_id': self.feed_id  # ✅ 包含镜头ID
        }


# =============================================================================
# PostProcessor - 后处理器类
# =============================================================================
class PostProcessor:
    """后处理器类"""
    
    def __init__(self, output_dir: str = './results'):
        """
        初始化后处理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.detection_history: List[DetectionResult] = []
        self.statistics: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)
    
    def get_history_count(self) -> int:
        with self._lock:
            return len(self.detection_history)
    
    def get_detection_history_copy(self) -> List[DetectionResult]:
        with self._lock:
            return list(self.detection_history)
    
    def add_detection(self, result_list: List[List], frame_id: Optional[int] = None,
                     feed_id: Optional[int] = None):
        """
        添加检测结果到历史记录
        
        这是后处理器最常被调用的方法，在 main.py 的每帧检测循环中，
        模型推理完成后立即调用此方法将结果存入。
        
        Args:
            result_list: 检测结果列表，来自 detector.inference_image() 的输出
                         每个元素格式: [class_name, confidence, x1, y1, x2, y2]
            frame_id: 当前帧的编号，由 main.py 中的 current_frame_id 计数器提供
            feed_id: 镜头ID (0-7)，表示来自 8 个画面中的哪一个
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            for result in result_list:
                det_result = DetectionResult(
                    class_name=result[0],
                    confidence=result[1],
                    bbox=(result[2], result[3], result[4], result[5]),
                    timestamp=timestamp,
                    frame_id=frame_id,
                    feed_id=feed_id  # ✅ 传入镜头ID
                )
                self.detection_history.append(det_result)
                self.statistics[result[0]] += 1
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取统计信息
        
        Returns:
            普通字典（非 defaultdict），键为类别名称，值为累计检测次数
        """
        with self._lock:
            return dict(self.statistics)
    
    def get_detection_summary(self) -> str:
        """获取检测摘要文本"""
        with self._lock:
            stats = dict(self.statistics)
        if not stats:
            return "未检测到目标"
        summary = "检测统计:\n"
        total = sum(stats.values())
        summary += f"总计: {total}\n"
        for class_name, count in stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            summary += f"{class_name}: {count} ({percentage:.1f}%)\n"
        return summary
    
    def save_image(self, image: np.ndarray, filename: Optional[str] = None, 
                   subfolder: str = 'images') -> str:
        """保存图像"""
        folder = os.path.join(self.output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.jpg"
        
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, image)
        return filepath
    
    def save_video(self, frames: List[np.ndarray], filename: Optional[str] = None,
                   fps: int = 30, subfolder: str = 'videos') -> str:
        """保存视频"""
        if not frames:
            return ""
        
        folder = os.path.join(self.output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.mp4"
        
        filepath = os.path.join(folder, filename)
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return filepath
    
    def export_json(self, filename: Optional[str] = None) -> str:
        """导出检测结果为JSON格式"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with self._lock:
            stats = dict(self.statistics)
            dets = [det.to_dict() for det in self.detection_history]
        data = {
            'statistics': stats,
            'total_detections': len(dets),
            'detections': dets,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def export_csv(self, filename: Optional[str] = None) -> str:
        """导出检测结果为CSV格式"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with self._lock:
            history_copy = list(self.detection_history)
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # ✅ 添加"镜头"列
            writer.writerow(['时间戳', '镜头', '帧ID', '类别', '置信度', 'X1', 'Y1', 'X2', 'Y2'])
            
            for det in history_copy:
                # ✅ 格式化镜头ID为"画面1"-"画面8"
                feed_name = f"画面{det.feed_id + 1}"
                writer.writerow([
                    det.timestamp,
                    feed_name,
                    det.frame_id if det.frame_id is not None else '',
                    det.class_name,
                    det.confidence,
                    det.bbox[0], det.bbox[1],
                    det.bbox[2], det.bbox[3]
                ])
        
        return filepath
    
    def export_report(self, filename: Optional[str] = None) -> str:
        """导出文本报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with self._lock:
            history_copy = list(self.detection_history)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("皮带传送带锚杆检测报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(self.get_detection_summary() + "\n")
            
            f.write("-" * 50 + "\n")
            f.write("详细检测记录:\n")
            f.write("-" * 50 + "\n")
            
            for i, det in enumerate(history_copy, 1):
                f.write(f"\n检测 #{i}:\n")
                f.write(f"  时间: {det.timestamp}\n")
                # ✅ 添加镜头信息
                f.write(f"  镜头: 画面{det.feed_id + 1}\n")
                if det.frame_id is not None:
                    f.write(f"  帧ID: {det.frame_id}\n")
                f.write(f"  类别: {det.class_name}\n")
                f.write(f"  置信度: {det.confidence:.2f}\n")
                f.write(f"  位置: ({det.bbox[0]}, {det.bbox[1]}) - ({det.bbox[2]}, {det.bbox[3]})\n")
        
        return filepath
    
    def clear_history(self):
        """清空历史记录"""
        with self._lock:
            self.detection_history.clear()
            self.statistics.clear()
    
    def get_recent_detections(self, count: int = 10) -> List[DetectionResult]:
        """获取最近的检测结果"""
        with self._lock:
            if count > 0:
                return list(self.detection_history[-count:])
            return list(self.detection_history)