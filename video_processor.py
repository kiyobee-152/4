# -*- coding: utf-8 -*-
"""
视频预处理模块
提供视频流的预处理功能：分辨率调整、帧率控制、图像增强等

本模块包含两个核心类：
  - VideoProcessor（视频处理器）：负责对每一帧图像进行预处理操作，包括分辨率缩放和
    图像增强（亮度、对比度、饱和度调节）。在 main.py 的检测循环中，每一帧在送入
    YOLOv5 模型推理之前都会先经过 VideoProcessor.process_frame() 处理。
    此外还提供了获取视频元信息、创建视频写入器等辅助功能。
  - FrameRateController（帧率控制器）：通过计算相邻帧之间的实际耗时与目标帧间隔的
    差值来决定是否需要 sleep 等待，从而将视频播放/检测的帧率稳定在目标值附近。
    在 main.py 的视频/摄像头检测循环末尾会调用 wait_if_needed() 进行帧率限制。

使用场景：
  - main.py 中 Ui_MainWindow 类在 __init__ 时创建 VideoProcessor 和 FrameRateController 实例
  - 在 start_camera()、start_video()、start_image() 方法中调用 process_frame() 进行帧预处理
  - GUI 中的"启用图像增强"复选框通过 toggle_enhancement() 控制 enable_enhancement 标志
  - 视频检测时会调用 get_video_info() 获取视频的 fps，并用该值初始化 FrameRateController

依赖关系：
  - cv2（OpenCV）：图像缩放、色彩空间转换、视频读写
  - numpy：数组运算（饱和度通道的浮点运算与裁剪）
"""
import cv2
import numpy as np
# Tuple: 用于标注返回值或参数中的元组类型（如分辨率 (width, height)）
# Optional: 表示参数可以为 None
# Callable: 预留的类型提示（当前未使用，但导入后可用于未来扩展回调函数参数）
from typing import Tuple, Optional, Callable


# =============================================================================
# VideoProcessor - 视频处理器类
# =============================================================================
# 职责：对输入的每一帧图像执行可选的预处理操作（分辨率调整和图像增强），
# 使其更适合后续的目标检测推理。所有预处理操作都是可配置的，默认情况下
# 不做任何处理（target_size=None, enable_enhancement=False），即透传原始帧。
class VideoProcessor:
    """视频处理器类"""
    
    def __init__(self, 
                 target_size: Optional[Tuple[int, int]] = None,
                 target_fps: Optional[int] = None,
                 enable_enhancement: bool = False,
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 saturation: float = 1.0):
        """
        初始化视频处理器
        
        Args:
            target_size: 目标分辨率 (width, height)，None表示保持原分辨率
            target_fps: 目标帧率，None表示保持原帧率
            enable_enhancement: 是否启用图像增强
            brightness: 亮度调整系数 (0.0-2.0)
            contrast: 对比度调整系数 (0.0-2.0)
            saturation: 饱和度调整系数 (0.0-2.0)
        """
        # 目标分辨率，格式为 (宽, 高)；如果为 None 则不缩放，保持原始尺寸
        # 在某些场景下可将高分辨率视频先缩小以提升检测速度
        self.target_size = target_size
        # 目标帧率（当前 VideoProcessor 类本身并未直接使用此属性，
        # 帧率控制由独立的 FrameRateController 类负责；此处保留用于记录配置信息）
        self.target_fps = target_fps
        # 图像增强总开关：False=不做增强直接透传，True=应用亮度/对比度/饱和度调节
        # 在 main.py 中通过 GUI 的"启用图像增强"复选框控制此标志
        self.enable_enhancement = enable_enhancement
        # 亮度调整系数：1.0=不变，<1.0=变暗，>1.0=变亮
        # 内部实现使用 cv2.convertScaleAbs 的 beta 参数来偏移像素值
        self.brightness = brightness
        # 对比度调整系数：1.0=不变，<1.0=降低对比度，>1.0=增强对比度
        # 内部实现使用 cv2.convertScaleAbs 的 alpha 参数来缩放像素值
        self.contrast = contrast
        # 饱和度调整系数：1.0=不变，<1.0=降低饱和度（偏灰），>1.0=增强饱和度（颜色更鲜艳）
        # 内部实现通���转换到 HSV 色彩空间，对 S（饱和度）通道进行缩放
        self.saturation = saturation
        self._clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像
        
        这是视频预处理的主入口方法，在 main.py 的每一帧检测循环中被调用。
        按顺序依次执行：分辨率调整 → 图像增强（如果启用）。
        
        处理流程：
        1. 如果设置了 target_size，使用双线性插值将帧缩放到目标分辨率
        2. 如果启用了图像增强（enable_enhancement=True），依次调节亮度、对比度、饱和度
        
        Args:
            frame: 输入帧，OpenCV 格式的 numpy 数组（BGR 色彩空间）
            
        Returns:
            处理后的帧，与输入格式相同（BGR numpy 数组）
        """
        # 调整分辨率：当 target_size 不为 None 时执行
        # cv2.INTER_LINEAR（双线性插值）在速度和质量之间取得了较好的平衡
        if self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 图像增强：仅在 enable_enhancement 标志为 True 时执行
        # 此标志可通过 GUI 的复选框动态切换
        if self.enable_enhancement:
            frame = self._enhance_image(frame)
        
        return frame
    
    def _enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """
        图像增强处理

        1. LAB 空间对亮度 L 做 CLAHE，提升暗部细节（整体偏暗时比线性加亮更明显）
        2. 亮度 / 对比度 / 饱和度：系数 ≠ 1.0 时才处理

        Args:
            frame: 输入图像（BGR 格式）

        Returns:
            增强后的图像（BGR 格式）
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        frame = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        # ---- 亮度调整 ----
        # 原理：output = alpha * input + beta
        # 这里 alpha=1.0（不缩放），beta 为偏移量
        # brightness=1.0 时 beta=0（不变）；brightness=1.5 时 beta=25（变亮）；
        # brightness=0.5 时 beta=-25（变暗）
        # convertScaleAbs 会自动将结果裁剪到 [0, 255] 范围并转为 uint8
        if self.brightness != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=int((self.brightness - 1.0) * 50))
        
        # ---- 对比度调整 ----
        # 原理：output = alpha * input + beta
        # 这里 alpha=self.contrast（缩放因子），beta=0（不偏移）
        # contrast>1.0 时增强对比度（亮的更亮、暗的更暗）；
        # contrast<1.0 时降低对比度（整体趋向于中间灰度）
        if self.contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=0)
        
        # ---- 饱和度调整 ----
        # 原理：将 BGR 图像转换到 HSV 色彩空间（H=色相, S=饱和度, V=明度），
        # 对 S 通道乘以饱和度系数后再转回 BGR
        # saturation>1.0 时颜色更鲜艳；saturation<1.0 时颜色偏灰/淡
        if self.saturation != 1.0:
            # BGR → HSV 色彩空间转换
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 转为 float32 以避免 uint8 乘法溢出
            # （uint8 最大值 255，乘以系数后可能超出范围）
            hsv = hsv.astype(np.float32)
            # 对 S 通道（索引 1）进行缩放
            hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
            # 裁剪到 [0, 255] 范围，防止溢出导致颜色异常
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            # 转回 uint8 类型（OpenCV 的 cvtColor 要求输入为 uint8）
            hsv = hsv.astype(np.uint8)
            # HSV → BGR 色彩空间转换回来
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return frame
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频文件的元信息
        
        使用 OpenCV 的 VideoCapture 读取视频文件头部信息，提取帧率、分辨率、
        总帧数和时长等关键参数。在 main.py 的 start_video() 方法中调用，
        用于在状态栏显示视频信息并设置 FrameRateController 的目标帧率。
        
        Args:
            video_path: 视频文件路径（支持 mp4、avi、flv 等格式）
            
        Returns:
            视频信息字典，包含以下键：
            - 'fps': 帧率（整数）
            - 'width': 视频宽度（像素）
            - 'height': 视频高度（像素）
            - 'frame_count': 总帧数
            - 'duration': 视频时长（秒，浮点数）
            如果视频无法打开则返回空字典 {}
        """
        # 创建 VideoCapture 对象尝试打开视频文件
        cap = cv2.VideoCapture(video_path)
        # 如果打开失败（文件不���在、格式不支持等），返回空字典
        if not cap.isOpened():
            return {}
        
        # 通过 CAP_PROP_* 属性获取视频元信息
        info = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),              # 帧率
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),     # 帧宽度（像素）
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),   # 帧高度（像素）
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), # 总帧数
            'duration': 0  # 秒，下面计算
        }
        
        # 计算视频时长：总帧数 / 帧率
        # 需要先确保 fps > 0，避免除零错误（某些特殊视频 fps 可能为 0）
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        # 释放 VideoCapture 资源（关闭文件句柄）
        cap.release()
        return info
    
    def create_video_writer(self, output_path: str, fps: int, size: Tuple[int, int], 
                           codec: str = 'mp4v') -> cv2.VideoWriter:
        """
        创建视频写入器
        
        封装了 OpenCV VideoWriter 的创建过程，用于将检测结果帧保存为视频文件。
        目前在 PostProcessor.save_video() 中有类似的逻辑，此方法提供了更灵活的
        编码器选择和参数配置。
        
        Args:
            output_path: 输出视频文件的保存路径
            fps: 目标帧率
            size: 视频尺寸，格式为 (width, height)
            codec: 视频编码格式的 FourCC 代码，默认 'mp4v'（MPEG-4 编码）
                   常用选项：
                   - 'mp4v': MPEG-4，兼容性好
                   - 'XVID': Xvid MPEG-4，压缩率较高
                   - 'MJPG': Motion JPEG，质量高但文件大
            
        Returns:
            cv2.VideoWriter 对象，可调用 .write(frame) 写入帧，使用完毕后需 .release()
        """
        # FourCC 是一个 4 字节的编码标识符，用于指定视频压缩编码格式
        # cv2.VideoWriter_fourcc(*codec) 将字符串 'mp4v' 展开为 'm','p','4','v' 传入
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(output_path, fourcc, fps, size)
    
    def set_enhancement_params(self, brightness: float = None, 
                               contrast: float = None, 
                               saturation: float = None):
        """
        动态设置图像增强参数
        
        允许在运行时调整亮度、对比度、饱和度系数。所有参数都会被限制在
        [0.0, 2.0] 的安全范围内，防止因极端值导致图像异常。
        传入 None 的参数不会被修改，保持当前值不变。
        
        可用于未来在 GUI 中添加亮度/对比度/饱和度滑块时的回调。
        
        Args:
            brightness: 亮度系数，None 表示不修改。0.0=全黑，1.0=原始，2.0=最亮
            contrast: 对比度系数，None 表示不修改。0.0=全灰，1.0=原始，2.0=最大对比度
            saturation: 饱和度系数，None 表示不修改。0.0=灰度图，1.0=原始，2.0=最大饱和度
        """
        if brightness is not None:
            # max(0.0, min(2.0, x)) 将值限制在 [0.0, 2.0] 范围内
            self.brightness = max(0.0, min(2.0, brightness))
        if contrast is not None:
            self.contrast = max(0.0, min(2.0, contrast))
        if saturation is not None:
            self.saturation = max(0.0, min(2.0, saturation))


# =============================================================================
# FrameRateController - 帧率控制器
# =============================================================================
# 职责：在视频逐帧检测的循环中控制处理速度，使帧率不超过目标值。
# 
# 工作原理：
#   假设目标帧率为 30fps，则每帧的理想间隔为 1/30 ≈ 0.033 秒。
#   每次调用 wait_if_needed() 时：
#   1. 计算距上一帧的实际耗时 elapsed
#   2. 如果 elapsed < frame_time（处理太快），则 sleep 补足差值
#   3. 如果 elapsed >= frame_time（处理已经够慢或更慢），则不等待
#   这样就能将帧率"钳制"在目标值以下，避免视频回放速度过快。
#
# 在 main.py 中的使用位置：
#   - start_camera() 和 start_video() 的 while 循环末尾调用 wait_if_needed()
#   - start_video() 开头通过 get_video_info() 获取视频原始 fps，
#     然后调用 set_fps() 同步到控制器
class FrameRateController:
    """帧率控制器"""
    
    def __init__(self, target_fps: int = 30):
        """
        初始化帧率控制器
        
        Args:
            target_fps: 目标帧率，即每秒最多处理的帧数。
                        默认 30fps，适用于大多数实时检测场景。
                        设为 0 或负数表示不限制帧率（尽可能快地处理）。
        """
        # 目标帧率值
        self.target_fps = target_fps
        # 每帧的理想时间间隔（秒）
        # 例如：target_fps=30 → frame_time=0.0333秒
        # 当 target_fps <= 0 时设为 0，表示不限制
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0
        # 上一帧处理完成的时间戳，初始为 None 表示尚未开始
        # 第一帧不需要等待（没有"上一帧"可以比较）
        self.last_time = None
        # 延迟导入 time 模块并保存引用，避免在高频调用的 wait_if_needed() 中重复导入
        import time
        self.time_module = time
    
    def wait_if_needed(self):
        """
        如果需要，等待以达到目标帧率
        
        核心逻辑：
        - 计算当前时间与上一帧���成时间的差值（elapsed）
        - 如果差值小于目标帧间隔（frame_time），说明处理速度太快，
          需要 sleep 补足剩余时间
        - 如果差值已经大于等于目标帧间隔，说明处理本身就够慢了，不需要额外等待
        
        注意：
        - frame_time <= 0 时直接返回，不做任何限制
        - 第一帧（last_time=None）不需要等待
        - sleep 的精度受操作系统调度影响，实际帧率可能有微小波动
        """
        # 如果 frame_time <= 0，表示不限制帧率，直接返回
        if self.frame_time <= 0:
            return
        
        # 获取当前时间戳
        current_time = self.time_module.time()
        if self.last_time is not None:
            # 计算距上一帧处理完成经过的时间
            elapsed = current_time - self.last_time
            # 如果实际耗时小于目标帧间隔，说明处理太快，需要等待
            # 例如：target_fps=30, frame_time=0.033s, elapsed=0.020s → sleep(0.013s)
            if elapsed < self.frame_time:
                self.time_module.sleep(self.frame_time - elapsed)
        
        # 更新时间戳为当前时刻（注意：在 sleep 之后再获取，保证下一帧的计时起点准确）
        self.last_time = self.time_module.time()
    
    def set_fps(self, fps: int):
        """
        动态设置目标帧率
        
        在 main.py 的 start_video() 方法中，读取视频文件的原始 fps 后调用此方法，
        使帧率控制器的目标帧率与视频源保持一致，确保视频以正常速度播放。
        
        例如：视频文件为 25fps → set_fps(25) → frame_time = 1/25 = 0.04秒/帧
        
        Args:
            fps: 新的目标帧率。设为 0 或负数表示不限制帧率。
        """
        self.target_fps = fps
        # 重新计算帧间隔；fps <= 0 时设为 0 表示不限制
        self.frame_time = 1.0 / fps if fps > 0 else 0