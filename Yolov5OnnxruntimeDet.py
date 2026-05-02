# =============================================================================
# Yolov5OnnxruntimeDet.py — YOLOv5 ONNX Runtime 检测器
# =============================================================================
#
# 本模块是整个皮带传送带锚杆检测系统中最底层的"检测引擎"，
# 直接与 ONNX Runtime 推理框架交互，完成从"原始图像"到"检测结果列表"的完整流水线。
#
# ---- 在系统架构中的位置 ----
#
#   main.py (GUI 主窗口)
#     └── model_interface.py (通用检测器接口 / 适配器层)
#           └── Yolov5OnnxruntimeDet.py  ← 本文件（底层 ONNX 推理引擎）
#                 └── yolov5_utils.py (预处理 letterbox / 后处理 NMS / 坐标还原)
#
#   model_interface.py 中的 YOLOv5ONNXDetector 类内部持有本类的实例，
#   将 inference_image() / draw_image() 等方法适配为 BaseDetector 统一接口。
#   但本文件也可脱离 GUI 独立运行（见文件末尾 __main__）。
#
# ---- 推理流水线概览（inference_image 内部 10 步）----
#
#   原始图像 (HWC, BGR, uint8, 任意尺寸)
#     ① letterbox ──→ (HWC, BGR, uint8, 640×640)
#     ② transpose+flip ──→ (CHW, RGB, uint8, 640×640)
#     ③ ascontiguousarray ──→ 内存连续化
#     ④ from_numpy+to ──→ torch.Tensor on device
#     ⑤ float()+/255 ──→ float32, [0,1]
#     ⑥ unsqueeze ──→ (1,3,640,640) batch 维度
#     ⑦ cpu().numpy() ──→ numpy, 送入 ONNX Runtime
#     ⑧ sess.run() ──→ 模型推理, 输出 (1,25200,5+nc)
#     ⑨ NMS ──→ 过滤+去重, 输出 (N,6)
#     ⑩ scale_coords+格式化 ──→ [name, conf, x1, y1, x2, y2] 列表
#
# ---- 为什么使用 ONNX 格式 ----
#   - ONNX (Open Neural Network Exchange) 是开放的模型交换格式
#   - 不需要完整的 PyTorch 训练框架，部署体积更小
#   - ONNX Runtime 支持 CPU / GPU / TensorRT 多种后端，推理速度快
#   - 适合生产环境部署
#
# ---- 依赖关系 ----
#   - cv2 (OpenCV): 图像读写、resize、绘图
#   - numpy: 数组操作
#   - onnxruntime: ONNX 模型推理引擎
#   - torch / torchvision: 张量操作、NMS 算子（后处理仍依赖 PyTorch）
#   - yolov5_utils: letterbox / non_max_suppression / scale_coords
# =============================================================================

# ---- 标准库 / 第三方库导入 ----
import cv2                      # OpenCV: 图像读写、resize、绘图
import numpy as np              # numpy: 数组操作
# 从 yolov5_utils 导入所有工具函数:
#   letterbox            — 预处理: 等比缩放 + 灰色填充
#   non_max_suppression  — NMS 后处理: 去除重叠框
#   scale_coords         — 坐标还原: 模型输入空间 → 原图空间
#   xywh2xyxy / xyxy2xywh — 坐标格式转换 (NMS 内部调用)
from yolov5_utils import *
# 注意: cv2 和 numpy 在第 1-2 行和第 4-5 行被重复导入了,
# 这是代码中的冗余, Python 会忽略重复导入, 不影响运行
import cv2
import numpy as np
# onnxruntime: 微软开源高性能推理引擎, 支持 ONNX 格式模型
# 支持 CPUExecutionProvider (CPU) 和 CUDAExecutionProvider (GPU) 两种后端
import onnxruntime
# torch: PyTorch 框架, 本文件中用于:
#   1. 检测 CUDA 可用性 (torch.cuda.is_available)
#   2. 预处理中的张量操作 (from_numpy / float / unsqueeze)
#   3. 后处理中 NMS 函数内部的张量运算和 torchvision.ops.nms 调用
import torch


# =============================================================================
# 颜色映射定义 - ✅ 新增
# =============================================================================
CLASS_COLORS = {
    'bolt': (0, 0, 255),              # 红色 - 锚杆
    'large_sized_coal': (0, 255, 255),  # 黄色 - 大块煤
    'Other_garbage': (255, 0, 0),     # 蓝色 - 其他垃圾
}

DEFAULT_BOX_COLOR = (0, 165, 255)    # 橙色 - 默认颜色（用于未定义的类别）


def get_box_color(class_name):
    """
    根据类别名称获取对应的检测框颜色
    
    Args:
        class_name: 类别名称（如 'bolt', 'large_sized_coal', 'Other_garbage'）
        
    Returns:
        BGR 颜色值元组 (B, G, R)
    """
    return CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)


# =============================================================================
# Yolov5OnnxruntimeDet — YOLOv5 ONNX Runtime 检测器类
# =============================================================================
# 封装了 YOLOv5 模型基于 ONNX Runtime 的完整推理流程。
#
# 职责:
#   1. 模型加载与管理: 加载 .onnx 模型文件, 管理 InferenceSession
#   2. 图像推理: 完整的 预处理 → 推理 → 后处理 流水线
#   3. 结果绘制: 在图像上绘制检测框和标签
#   4. 独立运行: 提供 imshow / start_camera / start_video 方法,
#      可脱离 GUI 在命令行中单独使用
#
# 关键属性:
#   sess (InferenceSession): ONNX Runtime 推理会话
#   names (list[str]): 类别名称列表, 如 ["bolt", "bulk"]
#   img_size (tuple): 模型输入尺寸 (640, 640)
#   confidence (float): 置信度阈值, 可由 GUI 实时调节
#   iou (float): IOU 阈值, 可由 GUI 实时调节
#   device (str): 运行设备 'cuda' 或 'cpu'
#   input_name (str): 模型输入节点名称 (通常为 "images")
#   output_name (str): 模型输出节点名称 (通常为 "output0")
# =============================================================================
class Yolov5OnnxruntimeDet(object):
    # =========================================================================
    # __init__ — 初始化检测器
    # =========================================================================
    # 初始化流程:
    #   1. 加载类别名称 (从文件或外部传入)
    #   2. 检测 CUDA 可用性, 选择推理后端
    #   3. 创建 ONNX Runtime InferenceSession, 加载模型权重
    #   4. 设置默认阈值 (confidence=0.45, iou=0.45)
    #   5. 获取模型输入/输出节点名称
    #   6. 执行一次预热推理 (warm up)
    #
    # Args:
    #   weights: ONNX 模型文件路径, 默认 './weights/yolov5s.onnx'
    #   names: 类别名称列表, 如 ["bolt", "bulk"]; None 时从默认文件加载
    def __init__(self, weights='./weights/yolov5s.onnx', names=None):
        # 类别名称列表: 将模型输出的类别索引 (0, 1, ...) 映射为可读名称
        # 在 inference_image() 中通过 self.names[int(cls)] 查表
        self.names = names
        # 外部未传入类别名称时, 从默认配置文件加载
        if names is None:
            self.load_labels('./weights/class_names.txt')
        # 模型输入尺寸: YOLOv5 训练时使用的固定分辨率
        # letterbox 预处理会将任意尺寸的图像缩放填充到此尺寸
        self.img_size = (640, 640)  # 训练权重的传入尺寸
        # ---- 设备与推理后端选择 ----
        # 检测当前环境是否有 NVIDIA GPU 且安装了 CUDA 版 PyTorch
        cuda = torch.cuda.is_available()
        # device 属性用于 from_numpy().to(device) 将张量放到正确设备
        self.device = 'cuda' if cuda else 'cpu'  # 根据pytorch是否支持gpu选择设备
        # 选择 ONNX Runtime 的执行提供者 (Execution Provider)
        # CUDA 可用时优先使用 GPU 加速, CPU 作为兜底
        # 列表顺序表示优先级: 前面的优先尝试
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else [
            'CPUExecutionProvider']  # 选择onnxruntime
        # ---- 加载 ONNX 模型 ----
        print('load onnx weights...')
        # 创建推理会话: 解析 .onnx 文件, 构建计算图, 分配内存
        # 这是整个初始化中最耗时的步骤
        self.sess = onnxruntime.InferenceSession(weights, providers=providers)  # 加载模型
        # ---- 默认推理阈值 ----
        # 在 model_interface.py 中通过 set_confidence() / set_iou() 由 GUI 滑块实时修改
        self.confidence = 0.45  # 置信度阈值: 低于此值的检测框在 NMS 中被过滤
        self.iou = 0.45         # IOU 阈值: 重叠度超过此值的框在 NMS 中被抑制
        # ---- 获取模型输入/输出节点名称 ----
        # ONNX 模型的每个输入/输出节点都有唯一名称标识
        # sess.run() 需要通过名称指定数据的流向
        # 典型值: output_name="output0" (形状 1×25200×(5+nc))
        #         input_name="images"   (形状 1×3×640×640)
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        # ---- 模型预热 (Warm Up) ----
        # 首次推理时 ONNX Runtime 需要做额外初始化 (JIT 编译、内存池分配、
        # CUDA kernel 编译等), 导致首次耗时远大于后续推理。
        # 用一张 300×300 的全黑假图执行一次推理, 让初始化在正式检测前完成。
        #warm up
        self.inference_image(np.zeros((300,300,3), dtype=np.uint8))
        print('weights loaded!')

    # =========================================================================
    # load_labels — 从文本文件加载类别名称列表
    # =========================================================================
    # 文件格式: 每行一个类别名, 行号 (从0开始) = 模型中的类别索引
    #   bolt    ← 索引 0
    #   bulk    ← 索引 1
    #
    # Args:
    #   file: 类别名称文件路径, 如 './weights/class_names.txt'
    def load_labels(self, file):
        with open(file, 'r') as f:
            # rstrip('\n'): 去除文件末尾可能存在的空行换行符
            # split('\n'): 按换行符分割为列表, 如 ["bolt", "bulk"]
            self.names = f.read().rstrip('\n').split('\n')

    # =========================================================================
    # inference_image — 对单张图像执行完整推理
    # =========================================================================
    # 这是本类最核心的方法, 包含从原始图像到检测结果的全部 10 个处理步骤。
    # 在 main.py 的检测循环中, 每帧调用一次。
    #
    # 数据变换全过程 (以 1376×776 原图为例):
    #
    #   原始图像 numpy (776,1376,3) HWC BGR uint8
    #     │ ① letterbox
    #     ▼
    #   numpy (640,640,3) HWC BGR uint8
    #     │ ② transpose+[::-1]
    #     ▼
    #   numpy (3,640,640) CHW RGB uint8
    #     │ ③ ascontiguousarray (内存连续化)
    #     │ ④ from_numpy+to(device)
    #     ▼
    #   torch.Tensor (3,640,640) CHW RGB uint8
    #     │ ⑤ float() + /255
    #     ▼
    #   torch.Tensor (3,640,640) CHW RGB float32 [0,1]
    #     │ ⑥ [None] 扩展 batch 维
    #     ▼
    #   torch.Tensor (1,3,640,640) BCHW RGB float32
    #     │ ⑦ cpu().numpy()
    #     ▼
    #   numpy (1,3,640,640) BCHW RGB float32
    #     │ ⑧ sess.run() ONNX 推理
    #     ▼
    #   torch.Tensor (1,25200,7) 原始预测
    #     │ ⑨ non_max_suppression
    #     ▼
    #   torch.Tensor (N,6) [x1,y1,x2,y2,conf,cls] (640×640 坐标空间)
    #     │ ⑩ scale_coords + 格式化
    #     ▼
    #   list[ [name, conf, x1, y1, x2, y2], ... ] (原图坐标空间)
    #
    # Args:
    #   image: 原始输入图像, numpy 数组, HWC 格式, BGR 色彩, uint8, 任意尺寸
    #
    # Returns:
    #   result_list: 检测结果列表, 每个元素为:
    #     [class_name(str), confidence(float), x1(int), y1(int), x2(int), y2(int)]
    #     坐标为原始图像像素空间中的整数值
    #     未检测到目标时返回空列表 []
    def inference_image(self, image):
        # ---- ① letterbox 等比缩放 + 灰色填充 ----
        # 将任意尺寸图像缩放到 640×640, 保持宽高比, 不足部分填充 (114,114,114) 灰色
        # stride=64: 输出尺寸需为 64 的整数倍 (YOLOv5 下采样步长要求)
        # auto=False: 不使用最小矩形, 确保严格输出 640×640
        # [0]: letterbox 返回 (img, ratio, pad), 只取图像
        img = letterbox(image, self.img_size, stride=64, auto=False)[0]
        # ---- ② 维度转换 + 色彩空间转换 ----
        # transpose((2,0,1)): HWC (640,640,3) → CHW (3,640,640)
        #   HWC 是 OpenCV/numpy 标准; CHW 是 PyTorch/ONNX 模型标准输入格式
        # [::-1]: 反转通道维, BGR [B,G,R] → RGB [R,G,B]
        #   OpenCV 默认 BGR; YOLOv5 训练时使用 RGB
        img = img.transpose((2, 0, 1))[::-1]  # HWC 转 CHW，BGR 转 RGB
        # ---- ③ 内存连续化 ----
        # [::-1] 切片产生非连续内存视图 (stride 为负),
        # torch.from_numpy 要求内存必须连续, 故需要 ascontiguousarray
        img = np.ascontiguousarray(img)
        # ---- ④ 转为 PyTorch 张量并放置到���备 ----
        # from_numpy: numpy → torch.Tensor
        # .to(self.device): 放到 GPU 或 CPU
        img = torch.from_numpy(img).to(self.device)
        # ---- ⑤ 类型转换 + 归一化 ----
        # 原始 uint8 (0-255) → float32 (0.0-1.0)
        # 这与 YOLOv5 训练时的预处理一致
        img = img.float()
        img /= 255
        # ---- ⑥ 扩展 batch 维度 ----
        # 当前形状 (3,640,640), 模型需要 (batch,3,640,640)
        # img[None] 等价于 img.unsqueeze(0), 在最前面加一维
        # 注: if len(img.shape) 条件恒为 True (shape 元组长度>0),
        #     原意可能是 if len(img.shape) == 3, 但不影响正确性
        if len(img.shape):
            img = img[None]

        # =====================================================================
        # 第二阶段: ONNX Runtime 推理
        # =====================================================================
        # ---- ⑦ 转回 numpy 送入 ONNX Runtime ----
        # ONNX Runtime 输入必须为 numpy 数组 (不接受 torch 张量)
        # .cpu(): 若在 GPU 上先移回 CPU (numpy 不支持 GPU 内存)
        # .numpy(): torch.Tensor → numpy.ndarray
        img = img.cpu().numpy()  # 传入cpu并转成numpy格式
        # ---- ⑧ ONNX 模型推理 ----
        # sess.run([输出名], {输入名: 数据}):
        #   参数1: 要获取的输出节点名称列表
        #   参数2: 输入数据字典
        #   返回: 列表, 每个元素对应一个输出节点
        # [0]: 取第一个 (也是唯一一个) 输出
        # 输出形状: (1, 25200, 5+nc), 25200 = 所有尺度 anchor 总数
        # 每个 anchor: [cx, cy, w, h, objectness, cls0_conf, cls1_conf, ...]
        # 用 torch.tensor() 包装, 因为后续 NMS 函数需要 torch 张量
        pred_onnx = torch.tensor(self.sess.run([self.output_name], {self.input_name: img})[0])

        # =====================================================================
        # 第三阶段: 后处理
        # =====================================================================
        # ---- ⑨ 非极大值抑制 (NMS) ----
        # 从数万个候选框中筛选出高置信度且不重叠的最终检测框
        # 返回: 列表, 每个元素为一张图的结果张量 (N,6): [x1,y1,x2,y2,conf,cls]
        pred = non_max_suppression(pred_onnx, self.confidence, self.iou, classes=None, agnostic=False, max_det=1000)

        # ---- ⑩ 坐标还原 + 结果格式化 ----
        result_list = []
        # 遍历 batch 中每张图 (本项目 batch=1, 只循环一次)
        for i, det in enumerate(pred):
            if len(det):
                # 坐标还原: 从模型输入空间 (640×640) 映射回原图像素空间
                # img.shape[2:] = (640, 640), image.shape = 原图 (H, W, C)
                # .round(): 四舍五入为整数像素坐标
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                # 遍历每个检测框, 格式化为统一列表格式
                # reversed: 从置信度低到高遍历, append 后列表中高置信度在前
                # *xyxy: 解包前 4 个值为列表 [x1, y1, x2, y2]
                # conf: 置信度; cls: 类别索引
                for *xyxy, conf, cls in reversed(det):  # 从末尾遍历
                    # view(1,4).view(-1): 确保 xyxy 为一维张量 (4,)
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1)
                    # 格式化: [类别名, 置信度(2位小数), x1, y1, x2, y2]
                    # self.names[int(cls)]: 类别索引 → 名称, 如 0 → "bolt"
                    result_list.append(
                        [self.names[int(cls)], round(float(conf), 2), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                         int(xyxy[3])])

        return result_list

    # =========================================================================
    # draw_image — 在图像上绘制所有检测框 (公开方法)
    # =========================================================================
    # 遍历 inference_image() 返回的结果列表, 对每个检测结果调用私有方法
    # __draw_image() 绘制矩形框和标签文字。
    #
    # 在 main.py 中通过 model_interface.py 的 draw_image() 间接调用,
    # 每帧检测完成后将检测框绘制到画面上。
    #
    # Args:
    #   result_list: inference_image() 的返回值
    #                每个元素: [class_name, confidence, x1, y1, x2, y2]
    #   opencv_img: 待绘制的图像, numpy BGR 格式 (原地修改)
    #
    # Returns:
    #   opencv_img: 绘制了检测框和标签的图像 (与输入同一对象)
    def draw_image(self, result_list, opencv_img):
        # 没有检测结果时直接返回原图
        if len(result_list) == 0:
            return opencv_img
        # 逐个绘制每个检测结果
        for result in result_list:
            class_name = result[0]
            # 标签文字格式: "类别名,置信度", 如 "bolt,0.92"
            label_text = class_name + ',' + str(result[1])
            # ✅ 根据类别获取对应颜色
            box_color = get_box_color(class_name)
            # 调用私有方法绘制单个框, 传入坐标 [x1,y1,x2,y2] 和标签，以及颜色
            opencv_img = self.__draw_image(opencv_img, [result[2], result[3], result[4], result[5]], label_text, box_color=box_color)
        return opencv_img

    # =========================================================================
    # __draw_image — 在图像上绘制单个检测框和标签 (私有方法)
    # =========================================================================
    # 修改自 YOLOv5 v6.0 官方代码 (utils/plots.py)。
    #
    # 绘制效果:
    #   ┌─────────────────┐
    #   │ bolt,0.92 │      │  ← 标签背景 (灰色) + 白色文字
    #   ├───────────┘      │
    #   │                  │  ← 蓝色矩形框
    #   │    目标区域       │
    #   └──────────────────┘
    #
    # 标签位置自适应:
    #   - 框上方空间足够时 → 标签在框外上方 (outside=True)
    #   - 框靠近图像顶部时 → 标签在框内下方 (outside=False)
    #
    # Args:
    #   opencv_img: 图像, numpy BGR (原地修改)
    #   box: [xmin, ymin, xmax, ymax] 整数像素坐标
    #   label: 标签文字, 如 "bolt,0.92" (不支持中文, OpenCV 默认字体限制)
    #   line_width: 线宽, None 时根据图像尺寸自适应计算
    #   box_color: 检测框颜色 BGR, 默认 (255,0,0) 蓝色
    #   txt_box_color: 标签背景色 BGR, 默认 (200,200,200) 浅灰
    #   txt_color: 文字颜色 BGR, 默认 (255,255,255) 白色
    #
    # Returns:
    #   opencv_img: 绘制后的图像
    def __draw_image(self, opencv_img, box, label='', line_width=None, box_color=(255, 0, 0),
                     txt_box_color=(200, 200, 200),
                     txt_color=(255, 255, 255)):
        '''
        code modified yolov5-6.0
        Args:
            opencv_img:
            box: [xmin,ymin,xmax,ymax]
            label: text,not support chinese
            line_width: None
            box_color:
            txt_box_color:
            txt_color:

        Returns:opencv image with draw box and label
        '''
        # 自适应线宽: 根据图像尺寸自动计算, 大图粗线小图细线
        # sum(shape) ≈ 高+宽+通道, 乘 0.003 再取整, 最小为 2
        lw = line_width or max(round(sum(opencv_img.shape) / 2 * 0.003), 2)  # line width
        # 左上角 p1 和右下角 p2 (OpenCV 绘图函数需要的格式)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # 绘制检测框矩形; LINE_AA = 抗锯齿, 线条更平滑
        cv2.rectangle(opencv_img, p1, p2, box_color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            # 字体线宽: 比框线宽少 1, 最小为 1
            tf = max(lw - 1, 1)  # font thickness
            # 计算标签文字渲染后的像素宽高
            # font=0 即 FONT_HERSHEY_SIMPLEX; fontScale=lw/3 随图像大小自适应
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            # 判断标签应显示在框外 (上方) 还是框内 (下方)
            # 如果框上方空间 >= 文字高度 + 3px 边距, 标签放外面
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            # 计算标签背景框的另一个角点
            # outside=True: 向上偏移; outside=False: 向下偏移
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # 绘制标签背景填充矩形 (-1 表示填充, 非空心)
            cv2.rectangle(opencv_img, p1, p2, txt_box_color, -1, cv2.LINE_AA)  # filled
            # 绘制标签文字
            # outside=True: 文字在背景框内靠底部 (p1.y - 2)
            # outside=False: 文字在背景框内靠底部 (p1.y + h + 2)
            cv2.putText(opencv_img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
        return opencv_img


    # =========================================================================
    # imshow — 弹窗显示检测结果 (独立运行模式)
    # =========================================================================
    # 先绘制检测框, 再用 OpenCV imshow 弹窗显示, 按任意键关闭。
    # 仅用于 __main__ 独立测试, GUI 模式下不调用此方法。
    #
    # Args:
    #   result_list: inference_image() 的返回值
    #   opencv_img: ���始图像
    def imshow(self, result_list, opencv_img):
        # 有检测结果时先绘制框
        if len(result_list) > 0:
            opencv_img = self.draw_image(result_list, opencv_img)
        # 弹出 OpenCV 窗口
        cv2.imshow('result', opencv_img)
        # waitKey(0): 阻塞等待按键, 按任意键继续
        cv2.waitKey(0)

    # =========================================================================
    # start_camera — 摄像头实时检测 (独立运行模式, 无 GUI)
    # =========================================================================
    # 打开摄像头 → 循环 {读帧→推理→绘制→显示} → 按 'q' 退出。
    # 用于脱离 PyQt5 GUI 的独立测试场景。
    # GUI 模式下, 摄像头检测由 main.py 的 start_camera() 方法处理。
    #
    # Args:
    #   camera_index: 摄像头设备索引, 默认 0 (系统默认摄像头)
    def start_camera(self, camera_index=0):
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        # 获取摄像头元信息 (用于日志输出)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频帧宽度和高度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},width={},height={}".format(frame_fps, frame_width, frame_height))
        # 检测主循环
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 推理 + 绘制
            result_list = self.inference_image(frame)
            result_img = self.draw_image(result_list, frame)
            # 显示
            cv2.imshow('frame', result_img)
            # waitKey(1): 等待 1ms + 检测键盘; & 0xFF: 取低8位 (跨平台兼容)
            # ord('q') = 113, 按 q 退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放摄像头资源
        cap.release()
        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()

    # =========================================================================
    # start_video — 视频文件检测 (独立运行模式, 无 GUI)
    # =========================================================================
    # 与 start_camera 逻辑完全一致, 区别仅在于输入源为视频文件。
    # 同样用于独立测试, GUI 模式下由 main.py 的 start_video() 处理。
    #
    # Args:
    #   video_file: 视频文件路径, 如 'D:\\car.mp4'
    def start_video(self, video_file):
        # 打开视频文件
        cap = cv2.VideoCapture(video_file)
        # 获取视频元信息 (帧率、分辨率)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频帧宽度和高度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("video fps={},width={},height={}".format(frame_fps, frame_width, frame_height))
        # 检测主循环 (与 start_camera 相同)
        while True:
            ret, frame = cap.read()
            if not ret:
                # 视频播放完毕或读取错误
                break
            result_list = self.inference_image(frame)
            result_img = self.draw_image(result_list, frame)
            cv2.imshow('frame', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


# =============================================================================
# 程序入口 (独立运行模式)
# =============================================================================
# 当本文件作为脚本直接运行时 (python Yolov5OnnxruntimeDet.py) 执行以下代码。
# 用于开发调试和独立测试, 不依赖 PyQt5 GUI。
#
# 使用方法:
#   1. 修改 weights 路径为实际的 .onnx 模型文件
#   2. 修改 load_labels 路径为实际的类别名称文件
#   3. 修改 cv2.imread 路径为实际的测试图片
#   4. 运行脚本, 弹窗显示检测结果
#   (也可取消 start_video 行的注释来测试视频检测)
if __name__ == '__main__':
    # 创建检测器实例, 指定 ONNX 模型路径
    detector = Yolov5OnnxruntimeDet(weights=r'E:\ai\yolov5-6.0\pre-model\yolov5s.onnx')
    # 加载类别名称文件 (覆盖 __init__ 中默认加载的名称)
    detector.load_labels(r'E:\official-model\yolov8\labels.txt')
    # detector.start_video(r'D:\car.mp4')  # 取消注释可测试视频检测
    # 读取测试图片
    img = cv2.imread(r'E:\test.png')
    # 执行推理
    result_list = detector.inference_image(img)
    # 弹窗显示结果 (按任意键关闭)
    detector.imshow(result_list=result_list, opencv_img=img)