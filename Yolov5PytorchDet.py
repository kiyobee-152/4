# -*- coding: utf-8 -*-
# =============================================================================
# Yolov5PytorchDet.py — YOLOv5 PyTorch (.pt) 检测器
# =============================================================================
#
# 本模块通过 torch.hub 加载原生 YOLOv5 .pt 权重，实现与 Yolov5OnnxruntimeDet
# 完全兼容的推理接口，供 model_interface.py 的工厂函数统一调用。
#
# ---- 在系统架构中的位置 ----
#
#   main.py (GUI 主窗口)
#     └── model_interface.py (通用检测器接口 / 适配器层)
#           ├── Yolov5OnnxruntimeDet.py  (ONNX 推理引擎)
#           └── Yolov5PytorchDet.py  ← 本文件（PyTorch .pt 推理引擎）
#
# ---- 依赖关系 ----
#   - torch / torchvision: 模型加载与推理
#   - cv2 (OpenCV): 图像预处理
#   - numpy: 数组操作
#   首次使用时 torch.hub 会自动从 GitHub 下载 YOLOv5 源码（约几 MB）并缓存到
#   ~/.cache/torch/hub/，后续离线环境也可正常使用。
# =============================================================================

import cv2
import numpy as np


# =============================================================================
# YOLOv5PyTorchDetector — 原生 .pt 模型检测器
# =============================================================================
class YOLOv5PyTorchDetector:
    """
    YOLOv5 PyTorch (.pt) 模型检测器。

    通过 torch.hub.load('ultralytics/yolov5', 'custom', path=...) 加载模型，
    提供与 YOLOv5ONNXDetector / Yolov5OnnxruntimeDet 一致的接口：
      - inference_image(frame)  → result_list
      - draw_image(result_list, frame) → annotated_frame
      - set_confidence(conf)
      - set_iou(iou)
      - device  属性（'cuda' 或 'cpu'）
    """

    # ---- 颜色映射（与 ONNX 检测器保持一致）----
    _CLASS_COLORS = {
        'bolt': (0, 0, 255),              # 红色 - 锚杆
        'large_sized_coal': (0, 255, 255),  # 黄色 - 大块煤
        'Other_garbage': (255, 0, 0),     # 蓝色 - 其他垃圾
    }
    _DEFAULT_BOX_COLOR = (0, 165, 255)    # 橙色 - 默认

    def __init__(
        self,
        weights: str,
        names=None,
        conf_thres: float = 0.45,
        iou_thres: float = 0.45,
        device_preference: str = 'auto',
    ):
        """
        初始化检测器并加载模型。

        Args:
            weights: .pt 模型文件路径
            names:   类别名称列表（None 时从模型自动读取）
            conf_thres: 置信度阈值，默认 0.45
            iou_thres:  IOU 阈值，默认 0.45
            device_preference: 'auto' | 'gpu' | 'cpu'

        Raises:
            ImportError:  缺少 torch 依赖时抛出，含安装指引
            RuntimeError: 模型文件无效或 torch.hub 加载失败时抛出
        """
        self.weights = weights
        self.names = names or []
        self.confidence = conf_thres
        self.iou = iou_thres
        self.device_preference = (device_preference or 'auto').lower()
        self.device = 'cpu'
        self._model = None
        self.img_size = (640, 640)

        self.load_model()

    # =========================================================================
    # load_model — 加载 PyTorch 模型
    # =========================================================================
    def load_model(self):
        """
        加载 .pt 模型。

        执行步骤：
        1. 检测 torch 是否已安装，给出清晰错误提示
        2. 根据 device_preference 确定推理设备（GPU / CPU）
        3. 通过 torch.hub 加载 YOLOv5 模型
        4. 读取模型内置的类别名称列表（若外部未提供）
        5. 预热推理（300×300 虚拟图像，避免首帧延迟）
        """
        # ---- 1. 检查 torch 是否可用 ----
        try:
            import torch
        except ImportError:
            raise ImportError(
                "加载 .pt 模型需要安装 PyTorch。\n"
                "请运行: pip install torch torchvision\n"
                "（CUDA GPU 版本请参考 https://pytorch.org/get-started/locally/）"
            )

        # ---- 2. 确定推理设备 ----
        cuda_available = torch.cuda.is_available()
        if self.device_preference == 'gpu':
            if cuda_available:
                self.device = 'cuda'
            else:
                print("⚠️ 选择了 GPU，但当前环境不支持 CUDA，已自动回退到 CPU")
                self.device = 'cpu'
        elif self.device_preference == 'cpu':
            self.device = 'cpu'
        else:  # 'auto'
            self.device = 'cuda' if cuda_available else 'cpu'

        device_label = "GPU (CUDA)" if self.device == 'cuda' else "CPU"
        print(f"正在加载PyTorch模型: {self.weights}，设备: {device_label}")

        # ---- 3. 通过 torch.hub 加载模型 ----
        # torch.hub 首次运行时会从 GitHub 下载 ultralytics/yolov5 源码并缓存；
        # 后续离线环境直接使用缓存，不需要网络。
        try:
            self._model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=self.weights,
                device=self.device,
                force_reload=False,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"加载 .pt 模型失败: {exc}\n\n"
                "排查建议：\n"
                "  1. 确认已安装 PyTorch:  pip install torch torchvision\n"
                "  2. 首次加载需要访问 GitHub 以下载 YOLOv5 源码，\n"
                "     如网络受限可先在有网络的机器上运行一次以填充缓存\n"
                "  3. 确认模型文件路径正确且为有效的 YOLOv5 .pt 格式\n"
                f"     当前路径: {self.weights}"
            ) from exc

        # ---- 4. 同步置信度 / IOU 阈值 ----
        self._model.conf = self.confidence
        self._model.iou = self.iou

        # ---- 5. 读取类别名称 ----
        if not self.names:
            raw_names = getattr(self._model, 'names', None)
            if isinstance(raw_names, dict):
                # YOLOv5 新版返回 {0: 'class0', 1: 'class1', ...}
                self.names = [raw_names[i] for i in sorted(raw_names.keys())]
            elif raw_names is not None:
                self.names = list(raw_names)
            else:
                # 回退：从同目录的 class_names.txt 加载
                import os
                names_file = os.path.join(os.path.dirname(self.weights), 'class_names.txt')
                if os.path.exists(names_file):
                    with open(names_file, 'r', encoding='utf-8') as f:
                        self.names = [ln.strip() for ln in f.read().rstrip('\n').split('\n') if ln.strip()]
                else:
                    print(f"⚠️ 警告: 未在模型或 {names_file} 中找到类别名称，使用默认值")
                    # 顺序必须与模型训练时的类别索引一致：
                    # 0→Other_garbage, 1→bolt, 2→large_sized_coal
                    # （与 YOLOv5ONNXDetector 的默认值保持一致）
                    self.names = ['Other_garbage', 'bolt', 'large_sized_coal']

        print(f"✅ 已加载类别: {self.names}，共 {len(self.names)} 类")

        # ---- 6. 模型预热 ----
        # 使用与生产推理相同的输入尺寸（img_size），确保预热充分覆盖运行路径
        _dummy = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        self.inference_image(_dummy)
        print("模型加载完成!")

    # =========================================================================
    # inference_image — 单张图像完整推理
    # =========================================================================
    def inference_image(self, image: np.ndarray):
        """
        对单张 BGR 图像执行 YOLOv5 推理。

        Args:
            image: 原始输入图像，numpy 数组，HWC 格式，BGR 色彩，uint8

        Returns:
            result_list: 检测结果列表，每个元素为
                [class_name(str), confidence(float), x1(int), y1(int), x2(int), y2(int)]
                坐标为原始图像像素空间中的整数值。
                未检测到目标时返回空列表 []。
        """
        if self._model is None:
            return []

        # 每次推理前同步最新阈值（GUI 滑块可能已调整）
        self._model.conf = self.confidence
        self._model.iou = self.iou

        # YOLOv5 模型期望 RGB 输入；OpenCV 默认 BGR，需反转通道
        img_rgb = image[:, :, ::-1].copy()

        # 执行推理（size 参数指定最长边缩放目标，对应训练尺寸 640）
        results = self._model(img_rgb, size=self.img_size[0])

        # results.xyxy[0]: 第一张（也是唯一一张）图像的检测结果
        # 形状 (N, 6)：[x1, y1, x2, y2, confidence, class_id]，坐标已还原至原图空间
        detections = results.xyxy[0].cpu().numpy()

        result_list = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls_id = int(cls)
            if cls_id < len(self.names):
                cls_name = self.names[cls_id]
            else:
                print(f"⚠️ 警告: 类别ID {cls_id} 超出范围（共 {len(self.names)} 类）")
                cls_name = f"Unknown_class_{cls_id}"
            result_list.append([
                cls_name,
                round(float(conf), 2),
                int(x1), int(y1), int(x2), int(y2),
            ])

        return result_list

    # =========================================================================
    # draw_image — 在图像上绘制检测结果
    # =========================================================================
    def draw_image(self, result_list, opencv_img: np.ndarray) -> np.ndarray:
        """
        在图像上绘制所有检测框和标签。

        Args:
            result_list: inference_image() 的返回值
            opencv_img:  原始图像（BGR 格式，numpy 数组）

        Returns:
            绘制了检测框的图像
        """
        if len(result_list) == 0:
            return opencv_img

        for result in result_list:
            class_name = result[0]
            label_text = f"{class_name},{result[1]}"
            box_color = self._CLASS_COLORS.get(class_name, self._DEFAULT_BOX_COLOR)
            opencv_img = self._draw_box(
                opencv_img,
                [result[2], result[3], result[4], result[5]],
                label_text,
                box_color=box_color,
            )
        return opencv_img

    def _draw_box(
        self,
        img: np.ndarray,
        box,
        label: str = '',
        line_width=None,
        box_color=(255, 0, 0),
        txt_box_color=(200, 200, 200),
        txt_color=(255, 255, 255),
    ) -> np.ndarray:
        """绘制单个检测框和对应的标签（与 ONNX 检测器实现一致）。"""
        lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
        p1 = (int(box[0]), int(box[1]))
        p2_rect = (int(box[2]), int(box[3]))
        cv2.rectangle(img, p1, p2_rect, box_color, thickness=lw, lineType=cv2.LINE_AA)

        if label:
            tf = max(lw - 1, 1)
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = p1[1] - h - 3 >= 0
            p2_label = (p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3)
            cv2.rectangle(img, p1, p2_label, txt_box_color, -1, cv2.LINE_AA)
            cv2.putText(
                img, label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA,
            )
        return img

    # =========================================================================
    # 阈值动态设置（供 GUI 滑块实时调用）
    # =========================================================================
    def set_confidence(self, conf: float):
        """动态设置置信度阈值。"""
        self.confidence = conf
        if self._model is not None:
            self._model.conf = conf

    def set_iou(self, iou: float):
        """动态设置 IOU 阈值。"""
        self.iou = iou
        if self._model is not None:
            self._model.iou = iou
