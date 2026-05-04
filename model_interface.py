# -*- coding: utf-8 -*-
"""
通用模型接口模块
支持多种深度学习模型格式（ONNX、PyTorch等）

本模块采用"抽象基类 + 工厂模式"的设计思路：
  - BaseDetector（抽象基类）：定义了所有检测器必须遵守的统一接口，包括模型加载、
    图像推理、预处理、结果绘制等方法，使上层调用方无需关心底层使用的是哪种模型格式。
  - YOLOv5ONNXDetector（具体实现类）：继承 BaseDetector，实现了基于 ONNX Runtime
    的 YOLOv5 目标检测推理，包含完整的预处理→推理→NMS后处理→坐标还原流水线。
  - create_detector（工厂函数）：根据模型文件的扩展名自动识别模型类型并实例化对应
    的检测器，对外提供统一的创建入口。

依赖关系：
  - yolov5_utils.py 中的 letterbox、non_max_suppression、scale_coords 等工具函数
  - onnxruntime（ONNX 模型推理引擎）
  - torch / torchvision（张量操作、NMS 等）
"""

# ABC: 用于定义抽象基类的元类
# abstractmethod: 装饰器，标记子类必须实现的抽象方法
from abc import ABC, abstractmethod
import numpy as np
import cv2
# List, Tuple, Optional: 类型提示工具，增强代码可读性和 IDE 自动补全
from typing import List, Tuple, Optional, Dict


# =============================================================================
# 颜色映射定义
# =============================================================================
# ✅ 新增：类别颜色映射（BGR 格式）
CLASS_COLORS = {
    'bolt': (0, 0, 255),              # 红色 - 锚杆
    'large_sized_coal': (0, 255, 255),  # 黄色 - 大块煤
    'Other_garbage': (255, 0, 0),     # 蓝色 - 其他垃圾
}

DEFAULT_BOX_COLOR = (0, 165, 255)    # 橙色 - 默认颜色（用于未定义的类别）


def get_box_color(class_name: str) -> Tuple[int, int, int]:
    """
    根据类别名称获取对应的检测框颜色
    
    Args:
        class_name: 类别名称（如 'bolt', 'large_sized_coal', 'Other_garbage'）
        
    Returns:
        BGR 颜色值元组 (B, G, R)
    """
    return CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)


# =============================================================================
# BaseDetector - 检测器抽象基类
# =============================================================================
# 所有检测器（无论是 ONNX、PyTorch 还是未来扩展的其他格式）都必须继承此基类，
# 并实现 load_model() 和 inference_image() 两个抽象方法。
# 基类还提供了通用的预处理（preprocess）、检测框绘制（draw_image/_draw_box）
# 以及阈值动态设置（set_confidence/set_iou）等已实现的公共方法。
class BaseDetector(ABC):
    """检测器基类，定义通用接口"""
    
    def __init__(self, weights: str, names: Optional[List[str]] = None, conf_thres: float = 0.45, iou_thres: float = 0.45):
        """
        初始化检测器
        
        Args:
            weights: 模型权重文件路径
            names: 类别名称列表
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
        """
        # 保存模型权重文件路径，供子类在 load_model() 中使用
        self.weights = weights
        # 类别名称列表，如 ['bolt', 'bulk']；若未传入则初始化为空列表，
        # 子类可在 load_model() 中从 class_names.txt 文件加载
        self.names = names or []
        # 置信度阈值：推理结果中低于此值的检测框将被过滤掉
        self.confidence = conf_thres
        # IOU（交并比）阈值：NMS 中用于判断两个框是否重叠的阈值，
        # 重叠度超过此值的框会被抑制（去重）
        self.iou = iou_thres
        # YOLOv5 模型的标准输入尺寸，图像在送入模型前会被缩放到此尺寸
        self.img_size = (640, 640)  # 默认输入尺寸
    
    # ---- 抽象方法：子类必须实现 ----
        
    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def inference_image(self, image: np.ndarray) -> List[List]:
        """
        对单张图像进行推理
        
        Args:
            image: 输入图像 (numpy array, BGR格式)
            
        Returns:
            检测结果列表，每个元素格式: [class_name, confidence, x1, y1, x2, y2]
        """
        pass
    
    # ---- 已实现的公共方法 ----
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple, Tuple]:
        """
        图像预处理（通用）
        
        使用 letterbox 算法将任意尺寸的输入图像缩放至模型所需的 img_size（640×640），
        在保持原始宽高比的前提下，不足部分用灰色（114, 114, 114）填充。
        stride=64 表示填充后的尺寸需为 64 的整数倍（YOLOv5 网络下采样倍数）。
        auto=False 表示不自动调整填充方向，始终在右下方填充。
        
        Args:
            image: ���入图像
            
        Returns:
            处理后的图像, 缩放比例, 填充大小
        """
        # 延迟导入，避免模块级别的循环依赖
        from yolov5_utils import letterbox
        return letterbox(image, self.img_size, stride=64, auto=False)
    
    def draw_image(self, result_list: List[List], opencv_img: np.ndarray) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        遍历所有检测结果，为每个目标绘制矩形框和类别+置信度标签文字。
        
        Args:
            result_list: 检测结果列表，每个元素为 [class_name, conf, x1, y1, x2, y2]
            opencv_img: 原始图像（BGR 格式的 numpy 数组）
            
        Returns:
            绘制了检测框的图像
        """
        # 如果没有检测到目标，直接返回原图
        if len(result_list) == 0:
            return opencv_img
        
        # 遍历每一���检测结果
        for result in result_list:
            class_name = result[0]
            # 拼接标签文字，格式如 "bolt, 0.92"
            label_text = f"{class_name}, {result[1]:.2f}"
            # ✅ 根据类别获取对应颜色
            box_color = get_box_color(class_name)
            # 调用内部方法绘制单个检测框
            # result[2:6] 分别为 x1, y1, x2, y2（左上角和右下角坐标）
            opencv_img = self._draw_box(opencv_img, 
                                       [result[2], result[3], result[4], result[5]], 
                                       label_text,
                                       box_color=box_color)
        return opencv_img
    
    def _draw_box(self, img: np.ndarray, box: List[int], label: str = '', 
                  line_width: Optional[int] = None, 
                  box_color: Tuple[int, int, int] = (255, 0, 0),
                  txt_box_color: Tuple[int, int, int] = (200, 200, 200),
                  txt_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        绘制单个检测框和对应的标签文字
        
        Args:
            img: 目标图像
            box: 边界框坐标 [x1, y1, x2, y2]
            label: 标签文字（如 "bolt, 0.92"）
            line_width: 边框线宽，None 则根据图像尺寸自适应计算
            box_color: 检测框颜色，BGR 格式，默认蓝色 (255, 0, 0)
            txt_box_color: 标签背景色，默认浅灰色
            txt_color: 标签文字颜色，默认白色
            
        Returns:
            绘制后的图像
        """
        # 自适应线宽：根据图像尺寸计算，确保在不同分辨率下检测框都清晰可见
        # sum(img.shape) 为 高+宽+通道数 之和，乘以 0.003 再除以 2 得到合理线宽
        lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
        # p1: 左上角坐标, p2: 右下角坐标
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # 绘制矩形检测框，使用抗锯齿 LINE_AA 模式
        cv2.rectangle(img, p1, p2, box_color, thickness=lw, lineType=cv2.LINE_AA)
        
        # 如果有标签文字，则绘制标签背景和文字
        if label:
            # tf: 文字线宽/粗细
            tf = max(lw - 1, 1)
            # 获取文字的像素宽度 w 和高度 h，用于计算标签背景大小
            # fontFace=0 即 cv2.FONT_HERSHEY_SIMPLEX
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            # 判断标签应该放在检测框上方还是下方
            # 如果框上方有足够空间（至少 h+3 像素），则放在上方
            outside = p1[1] - h - 3 >= 0
            # 计算标签背景矩形的右下角坐标
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # 绘制填充矩形作为标签背景（thickness=-1 表示填充）
            cv2.rectangle(img, p1, p2, txt_box_color, -1, cv2.LINE_AA)
            # 绘制标签文字
            # 如果标签在上方，文字 y 坐标为 p1[1]-2（稍微上移避免遮挡）
            # 如果标签在下方，文字 y 坐标为 p1[1]+h+2（下移到背景矩形内）
            cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                       0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        return img
    
    def set_confidence(self, conf: float):
        """
        动态设置置信度阈值
        
        在 GUI 界面中用户拖动置信度滑块时会调用此方法，
        实时更新阈值，使下一帧检测立即使用新阈值。
        """
        self.confidence = conf
    
    def set_iou(self, iou: float):
        """
        动态设置 IOU 阈值
        
        在 GUI 界面中用户拖动 IOU 滑块时会调用此方法，
        IOU 越大，NMS 越宽松（允许更多重叠框保留）；越小越严格。
        """
        self.iou = iou


# =============================================================================
# YOLOv5ONNXDetector - 基于 ONNX Runtime 的 YOLOv5 检测器
# =============================================================================
# 这是 BaseDetector 的具体实现类，使用 ONNX Runtime 作为推理引擎加载并运行
# YOLOv5 模型。ONNX 是一种开放的神经网络交换格式，相比直接使用 PyTorch 推理，
# ONNX Runtime 通常具有更好的推理性能和更广泛的部署兼容性。
class YOLOv5ONNXDetector(BaseDetector):
    """YOLOv5 ONNX模型检测器"""
    
    def __init__(
        self,
        weights: str,
        names: Optional[List[str]] = None,
        conf_thres: float = 0.45,
        iou_thres: float = 0.45,
        device_preference: str = 'auto'
    ):
        # 调用父类构造函数，初始化 weights、names、confidence、iou、img_size
        super().__init__(weights, names, conf_thres, iou_thres)
        # ONNX Runtime 推理会话对象，在 load_model() 中初始化
        self.sess = None
        # 模型输入张量的名称（如 "images"），用于构建推理输入字典
        self.input_name = None
        # 模型输出张量的名称（如 "output"），用于获取推理输出
        self.output_name = None
        # 推理设备：'cuda'（GPU）或 'cpu'
        self.device = 'cpu'
        # 设备偏好：'auto'（自动）/ 'cpu' / 'gpu'
        self.device_preference = (device_preference or 'auto').lower()
        # 构造函数中直接调用 load_model()，确保实例化后模型即可使用
        self.load_model()
    
    def load_model(self):
        """
        加载 ONNX 模型
        
        执行步骤：
        1. 检测 CUDA 是否可用，选择 GPU 或 CPU 作为推理设备
        2. 使用 onnxruntime.InferenceSession 加载 .onnx 模型文件
        3. 获取模型的输入/输出张量名称
        4. 如果未指定类别名称，自动从模型同目录下的 class_names.txt 加载
        5. 用一张全零虚拟图像做一次预热推理（warm up），
           使 ONNX Runtime 完成内部优化和内存分配，避免首次正式推理时延迟过高
        """
        import onnxruntime
        import torch
        
        # 检测当前环境是否支持 CUDA（NVIDIA GPU 加速）
        cuda_available = torch.cuda.is_available()
        available_providers = onnxruntime.get_available_providers()
        cuda_provider_available = 'CUDAExecutionProvider' in available_providers
        gpu_available = cuda_available and cuda_provider_available

        if self.device_preference == 'gpu':
            if gpu_available:
                self.device = 'cuda'
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("⚠️ 选择了 GPU，但当前环境不可用，已自动回退到 CPU")
                self.device = 'cpu'
                providers = ['CPUExecutionProvider']
        elif self.device_preference == 'cpu':
            self.device = 'cpu'
            providers = ['CPUExecutionProvider']
        else:
            self.device = 'cuda' if gpu_available else 'cpu'
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu_available else ['CPUExecutionProvider']
        
        print(f'正在加载ONNX模型: {self.weights}，设备: {"GPU" if self.device == "cuda" else "CPU"}')
        # 创建 ONNX Runtime 推理会话，加载模型权重
        self.sess = onnxruntime.InferenceSession(self.weights, providers=providers)
        # 获取模型定义的输出节点名称（YOLOv5 通常只有一个输出）
        self.output_name = self.sess.get_outputs()[0].name
        # 获取模型定义的输入节点名称
        self.input_name = self.sess.get_inputs()[0].name
        
        # 如果构造时未提供类别名称列表，则尝试自动加载
        # 约定：class_names.txt 与模型文件放在同一目录下
        # ✅ 修改：支持新的3类模型
        if not self.names:
            import os
            names_file = os.path.join(os.path.dirname(self.weights), 'class_names.txt')
            if os.path.exists(names_file):
                with open(names_file, 'r', encoding='utf-8') as f:
                    # 去除末尾换行符后按行分割，得到类别名称列表
                    lines = f.read().rstrip('\n').split('\n')
                    # 过滤空行
                    self.names = [line.strip() for line in lines if line.strip()]
            else:
                print(f"⚠️ 警告: 未找到 class_names.txt，路径: {names_file}")
                # ✅ 默认使用新的3个类别
                self.names = ['Other_garbage', 'bolt', 'large_sized_coal']
        
        # ✅ 调试输出：检查加载的类别
        print(f"✅ 已加载类别: {self.names}")
        print(f"✅ 类别数量: {len(self.names)}")
        
        # 模型预热（Warm up）：用一张 300×300 的全零黑色图像做一次完整推理
        # 目的是让 ONNX Runtime 提前完成图优化、内存分配等初始化操作，
        # 从而避免后续首次正式推理时出现明显的延迟抖动
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.inference_image(dummy_img)
        print('模型加载完成!')
    
    def inference_image(self, image: np.ndarray) -> List[List]:
        """
        对单张图像进行完整的 YOLOv5 推理流水线
        
        完整处理流程：
        1. 预处理（letterbox 缩放+填充 → 通道转换 → 归一化 → 增加 batch 维度）
        2. ONNX Runtime 推理
        3. NMS（非极大值抑制）后处理
        4. 坐标还原（从 640×640 模型输入空间映射回原图尺寸）
        5. 组装结果列表
        
        Args:
            image: 输入的原始图像，numpy 数组，BGR 格式（OpenCV 默认）
            
        Returns:
            result_list: 检测结果列表，每个元素为
                [class_name(str), confidence(float), x1(int), y1(int), x2(int), y2(int)]
                其中 (x1,y1) 为左上角坐标，(x2,y2) 为右下角坐标，均为原图上的像素坐标
        """
        import torch
        from yolov5_utils import non_max_suppression, scale_coords
        
        # ===================== 第一步：预处理 =====================
        # letterbox 缩放：将原始图像等比例缩放到 640×640，不足部分用灰色填充
        # 返回值：img=缩放填充后的图像, ratio=缩放比例, pad=填充的像素数
        img, ratio, pad = self.preprocess(image)
        # 通道转换：
        #   transpose((2,0,1)) 将 HWC（高×宽×通道）转为 CHW（通道×高×宽），这是 PyTorch/ONNX 的标准格式
        #   [::-1] 将 BGR 通道顺序反转为 RGB（OpenCV 默认 BGR，而模型训练时使用 RGB）
        img = img.transpose((2, 0, 1))[::-1]  # HWC转CHW，BGR转RGB
        # 确保数组在内存中是连续存储的（transpose 和切片可能产生不连续的视图）
        # 这是后续 torch.from_numpy() 的要求
        img = np.ascontiguousarray(img)
        # 将 numpy 数组转为 PyTorch 张量，并移至指定设备（GPU 或 CPU）
        img = torch.from_numpy(img).to(self.device)
        # 将数据类型从 uint8（0-255整数）转为 float32（浮点数）
        img = img.float()
        # 归一化：像素值从 [0, 255] 缩放到 [0.0, 1.0]，这是 YOLOv5 模型的输入要求
        img /= 255
        # 增加 batch 维度：从 [C, H, W] 变为 [1, C, H, W]
        # ONNX 模型的输入���求是 4 维张量（batch_size, channels, height, width）
        if len(img.shape) == 3:
            img = img[None]
        
        # ===================== 第二步：ONNX Runtime 推理 =====================
        # 将张量从 GPU 移回 CPU 并转为 numpy（ONNX Runtime 输入需要 numpy 格式）
        img_np = img.cpu().numpy()
        # 执行 ONNX 模型推理
        # sess.run() 参数说明：
        #   第一个参数 [self.output_name]: 需要获取的输出节点名称列表
        #   第二个参数 {self.input_name: img_np}: 输入字典，键为输入节点名，值为输入数据
        # 返回值是一个列表，[0] 取第一个（也是唯一一个）输出
        # 输出形状为 [1, N, 7]，N 为候选检测框数量，7 = [x, y, w, h, objectness, cls1_conf, cls2_conf]
        pred_onnx = torch.tensor(self.sess.run([self.output_name], {self.input_name: img_np})[0])
        
        # ===================== 第三步：NMS（非极大值抑制）后处理 =====================
        # non_max_suppression 的作用：
        #   1. 过滤掉置信度低于 self.confidence 阈值的候选框
        #   2. 对剩余的框按类别进行 NMS，去除 IOU 超过 self.iou 阈值的重叠框
        #   3. 限制每张图像最多保留 max_det=1000 个检测结果
        # classes=None 表示不按特定类别筛选
        # agnostic=False 表示 NMS 按类别分别进行（不同类别的框不互相抑制）
        # 返回值 pred 是一个列表，每个元素对应一张图像的检测结果张量，
        # 每个张量形状为 [M, 6]，M 为检测框数量，6 = [x1, y1, x2, y2, confidence, class_id]
        pred = non_max_suppression(pred_onnx, self.confidence, self.iou, classes=None, agnostic=False, max_det=1000)
        
        # ===================== 第四步：坐标还原与结果组装 =====================
        result_list = []
        # 遍历每张图像的检测结果（这里 batch_size=1，所以只循环一次）
        for i, det in enumerate(pred):
            if len(det):
                # scale_coords: 将检测框坐标从模型输入尺寸（640×640 带填充）
                # 映射回原始图像的实际尺寸
                # img.shape[2:] 为模型输入的 [H, W]（即 [640, 640]）
                # image.shape 为原始图像的 [H, W, C]
                # .round() 将坐标四舍五入为整数像素值
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                # 遍历每个检测框（reversed 从末尾开始遍历，不影响结果，仅为兼容旧代码习惯）
                # *xyxy: 解包前4个值为列表 [x1, y1, x2, y2]
                # conf: 检测置信度
                # cls: 类别 ID（整数索引）
                for *xyxy, conf, cls in reversed(det):
                    # 将 xyxy 坐标从列表转为一维张量，便于通过索引取值
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1)
                    # ✅ 修改：更好的类别映射，避免 class_2 问题
                    cls_id = int(cls)
                    if cls_id < len(self.names):
                        cls_name = self.names[cls_id]
                    else:
                        # ⚠️ 类别ID超出范围 - 这表示 class_names.txt 没有正确加载
                        print(f"⚠️ 警告: 类别ID {cls_id} 超出范围 (只有 {len(self.names)} 个类别)")
                        cls_name = f"Unknown_class_{cls_id}"
                    
                    # 将当前检测框的完整信息组装为列表并追加到结果列表
                    # 格式：[类别名称, 置信度(保留2位小数), 左上角x, 左上角y, 右下角x, 右下角y]
                    result_list.append([
                        cls_name,
                        round(float(conf), 2),
                        int(xyxy[0]), int(xyxy[1]),
                        int(xyxy[2]), int(xyxy[3])
                    ])
        
        return result_list


# =============================================================================
# create_detector - 检测器工厂函数
# =============================================================================
# 这是外部代码创建检测器的唯一入口。它封装了检测器的实例化逻辑，
# 调用方只需提供模型权重文件路径，无需关心底层使用哪个检测器类。
# 在 main.py 的 cb_weights_changed() 方法中，当用户在 GUI 下拉框中切换模型时，
# 会调用此函数来动态加载新模型。
def create_detector(weights: str, model_type: str = 'auto', **kwargs):
    """
    工厂函数：根据模型文件创建对应的检测器

    设计意图：
    - 对外隐藏具体的检测器实现类，调用方通过此函数统一创建
    - 支持自动识别模型类型（根据文件扩展名），也可手动指定
    - 通过 **kwargs 透传参数（如 conf_thres、iou_thres）给具体检测器构造函数
    - 未来扩展新的模型格式时，只需在此函数中添加分支即可

    Args:
        weights: 模型权重文件路径
            - ONNX 格式：如 './weights/yolov5s.onnx'
            - PyTorch 格式：如 './weights/yolov5s.pt'
        model_type: 模型类型，可选值：
            - 'auto': 自动根据文件扩展名判断（默认）
            - 'onnx': 强制使用 ONNX 检测器
            - 'pytorch': 强制使用 PyTorch (.pt) 检测器
        **kwargs: 传递给检测器构造函数的额外参数，支持：
            - names (List[str]): 类别名称列表
            - conf_thres (float): 置信度阈值
            - iou_thres (float): IOU 阈值
            - device_preference (str): 'auto' | 'gpu' | 'cpu'

    Returns:
        检测器实例（YOLOv5ONNXDetector 或 YOLOv5PyTorchDetector）

    Raises:
        ValueError: 当指定了不支持的模型类型时
        ImportError / RuntimeError: 加载 .pt 模型时缺少依赖或文件无效
    """
    import os

    if model_type == 'auto':
        # 自动检测模型类型：通过文件扩展名判断
        # os.path.splitext() 将路径拆分为 (去掉扩展名的部分, 扩展名)
        # 例如 './weights/yolov5s.onnx' → ('.onnx')
        ext = os.path.splitext(weights)[1].lower()
        if ext == '.onnx':
            model_type = 'onnx'
        elif ext in ['.pt', '.pth']:
            # .pt 和 .pth 都是 PyTorch 的常见权重文件扩展名
            model_type = 'pytorch'
        else:
            # 对于无法识别的扩展名，默认按 ONNX 格式处理
            model_type = 'onnx'  # 默认

    # 根据确定的模型类型实例化对应的检测器
    if model_type == 'onnx':
        # 创建并返回 ONNX 检测器实例
        # **kwargs 会将 conf_thres、iou_thres 等参数传入构造函数
        return YOLOv5ONNXDetector(weights, **kwargs)
    elif model_type == 'pytorch':
        # 延迟导入：仅在实际需要加载 .pt 模型时引入，
        # 避免没有安装 torch 的环境在导入 model_interface 时就崩溃
        try:
            from Yolov5PytorchDet import YOLOv5PyTorchDetector
        except ImportError as exc:
            raise ImportError(
                "加载 .pt 模型失败：无法导入 Yolov5PytorchDet 模块。\n"
                "请确认 Yolov5PytorchDet.py 文件存在，\n"
                "并已安装 PyTorch: pip install torch torchvision"
            ) from exc
        return YOLOv5PyTorchDetector(weights, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")