import torchvision
import cv2
import numpy as np
import torch

'''
工具包,所有的预处理和后处理函数都在这

本模块是 YOLOv5 目标检测流水线的核心工具函数集合，源自 ultralytics/yolov5 官方仓库，
经裁剪后仅保留本项目（皮带传送带锚杆检测系统）所需的关键函数。

本模块包含三大类功能：

1. 坐标格式转换（xywh2xyxy / xyxy2xywh）
   - YOLOv5 模型内部使用 [中心x, 中心y, 宽, 高] 格式表示边界框
   - 绘图和显示使用 [左上角x, 左上角y, 右下角x, 右下角y] 格式
   - 两个函数在这两种格式之间互转

2. 预处理（letterbox）
   - 将任意尺寸的输入图像缩放至模型所需的固定尺寸（如 640×640）
   - 保持原始宽高比不变，不足部分用灰色像素填充
   - 这是送入模型推理前的必须步骤

3. 后处理（non_max_suppression / clip_coords / scale_coords）
   - NMS（非极大值抑制）：从模型输出的大量候选框中去除重叠框，保留最佳检测结果
   - 坐标裁剪：确保边界框不超出图像范围
   - 坐标还原：将模型输出的坐标从 640×640 输入空间映射回原始图像尺寸

数据流向：
  原始图像 → [letterbox 预处理] → 640×640 图像 → [模型推理] → 原始预测
  → [non_max_suppression 后处理] → 过滤后的检测框（640×640 坐标空间）
  → [scale_coords 坐标还原] → 最终检测框（原始图像坐标空间）

被调用位置：
  - model_interface.py 中的 BaseDetector.preprocess() 调用 letterbox
  - model_interface.py 中的 YOLOv5ONNXDetector.inference_image() 调用
    non_max_suppression 和 scale_coords

依赖关系：
  - torch / torchvision：张量运算、NMS 算子
  - cv2（OpenCV）：图像缩放、边界填充
  - numpy：数组操作
'''

# =============================================================================
# 坐标格式转换函数
# =============================================================================

'''坐标转换'''
# 中心点xy,宽高wh转换到左上角xy和右下角xy
def xywh2xyxy(x):
    """
    将边界框从 [中心x, 中心y, 宽, 高] 格式转换为 [左上角x, 左上角y, 右下角x, 右下角y] 格式。

    这是 YOLOv5 后处理中的关键步骤。模型内部使用 xywh 格式（以中心点为锚点），
    但绘图函数（如 cv2.rectangle）和 NMS 函数（torchvision.ops.nms）都需要 xyxy 格式。

    转换公式：
      x1 = cx - w/2    （中心x 减去 半宽 = 左边界）
      y1 = cy - h/2    （中心y 减去 半高 = 上边界）
      x2 = cx + w/2    （中心x 加上 半宽 = 右边界）
      y2 = cy + h/2    （中心y 加上 半高 = 下边界）

    图示：
         (x1,y1)─────────────┐
           │                 │
           │    (cx,cy) ●    │  h
           │                 │
           └─────────────(x2,y2)
                   w

    Args:
        x: 形状为 (N, 4) 的张量或数组，每行为 [cx, cy, w, h]
           支持 torch.Tensor 和 numpy.ndarray 两种类型

    Returns:
        y: 形状相同的 (N, 4) 张量或数组，每行为 [x1, y1, x2, y2]
           返回类型与输入类型一致
    """
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # 深拷贝输入数据，避免修改原始数据
    # torch.Tensor 使用 .clone()，numpy 数组使用 np.copy()
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 左上角 x = 中心x - 宽度/2
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    # 左上角 y = 中心y - 高度/2
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    # 右下角 x = 中心x + 宽度/2
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    # 右下角 y = 中心y + 高度/2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# 左上角xy,右下角xy转换至左上角xy,宽高wh
def xyxy2xywh(x):
    """
    将边界框从 [左上角x, 左上角y, 右下角x, 右下角y] 格式转换为 [中心x, 中心y, 宽, 高] 格式。

    这是 xywh2xyxy 的逆运算。本项目中当前未直接使用，但保留用于可能的扩展需求
    （如将检测结果保存为 YOLO 标注格式时需要此转换）。

    转换公式：
      cx = (x1 + x2) / 2    （左右边界的平均值 = 中心x）
      cy = (y1 + y2) / 2    （上下边界的平均值 = 中心y）
      w  = x2 - x1          （右边界 - 左边界 = 宽度）
      h  = y2 - y1          （下边界 - 上边界 = 高度）

    Args:
        x: 形状为 (N, 4) 的张量或数组，每行为 [x1, y1, x2, y2]

    Returns:
        y: 形状相同的 (N, 4) 张量或数组，每行为 [cx, cy, w, h]
    """
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    # 深拷贝，避免修改原始数据
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 中心 x = (左上角x + 右下角x) / 2
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    # 中心 y = (左上角y + 右下角y) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    # 宽度 = 右下角x - 左上角x
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    # 高度 = 右下角y - 左上角y
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# =============================================================================
# 预处理函数
# =============================================================================

'''--------预处理--------'''
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Letterbox 图像预处理：等比例缩放 + 边界填充。

    YOLOv5 要求输入图像为固定尺寸（默认 640×640），且尺寸必须是 stride（下采样步长）
    的整数倍。但原始图像的尺寸和宽高比各不相同，直接 resize 会导致目标变形。
    letterbox 的解决方案是：
      1. 先等比例缩放图像（保持宽高比），使最长边恰好等于目标尺寸
      2. 在较短边的两侧对称填充灰色像素（114, 114, 114），补足到目标尺寸

    完整处理流程示意（以 1376×776 原图缩放到 640×640 为例）：

      原图 (1376×776)                     缩放后 (640×361)
      ┌──────────────┐                   ┌──────────┐
      │              │  等比例缩放 r=0.465│          │
      │              │  ──────────────→  │          │
      │  1376 × 776  │                   │ 640×361  │
      └──────────────┘                   └──────────┘

                                          填充后 (640×640)
                                         ┌──────────┐
                                         │▓▓填充▓▓▓▓│ ← 上方填充 139px
                                         │          │
                                         │ 640×361  │ ← 原始内容
                                         │          │
                                         │▓▓填充▓▓▓▓│ ← 下方填充 140px
                                         └──────────┘

    在 model_interface.py 中由 BaseDetector.preprocess() 调用，
    参数为 stride=64, auto=False（不使用最小矩形，确保输出为固定的 640×640）。

    Args:
        im: 输入图像，numpy 数组，HWC 格式（高×宽×通道），BGR 色彩空间
        new_shape: 目标尺寸，(高, 宽) 元组或整数（整数时高宽相同）。默认 (640, 640)
        color: 填充颜色，BGR 格式的元组。默认 (114, 114, 114) 即中灰色，
               这是 YOLOv5 训练时使用的标准填充颜色
        auto: 是否使用"最小矩形"模式。
              True: 填充量取 stride 的余数（最小填充，输出尺寸可能不等于 new_shape）
              False: 填充至完整的 new_shape 尺寸（本项目使用 False）
        scaleFill: 是否直接拉伸到目标尺寸（不保持宽高比）。默认 False，一般不使用
        scaleup: 是否允许放大。False 时只缩小不放大（小图保持原尺寸，用于验证时提升 mAP）
        stride: 网络下采样步长，填充后的尺寸需为 stride 的整数倍。
                YOLOv5 的 P5 模型最大下采样 32 倍，本项目中 model_interface.py 传入 64

    Returns:
        im: 缩放并填充后的图像，numpy 数组
        ratio: ���放比例元组 (r, r)，高和宽使用相同的缩放比
        (dw, dh): 宽度和高度方向上单侧的填充量（像素，浮点数），
                  用于 scale_coords 中还原坐标时减去填充偏移
    """
    # 在满足跨步约束的同时调整图像大小和填充图像
    # shape[0]=高度, shape[1]=宽度（numpy/OpenCV 的 HWC 格式）
    shape = im.shape[:2]  # 当前形状[高度，宽度]  #注意顺序
    # print(f'图片尺寸:{shape}')
    # 如果 new_shape 是单个整数，扩展为 (h, w) 元组
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # new_shape:(640, 640)

    # ---- 第一步：计算等比例缩放比例 r ----
    # 分别计算高度和宽度的缩放比例，取较小值以确保缩放后的图像完全在目标尺寸内
    # 例如：原图 1376×776，目标 640×640
    #   高度比 = 640/776 = 0.824
    #   宽度比 = 640/1376 = 0.465
    #   r = min(0.824, 0.465) = 0.465（以宽度为瓶颈）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # 找出图片的(高宽)最小的比例，[0]是高，[1]是宽
    if not scaleup:  # 只缩小，不放大（为了更好的 val mAP）默认跳过
        # 当 r > 1.0 时（原图比目标小），将 r 限制为 1.0（不放大）
        r = min(r, 1.0) # # 若有大于1的则用1比例，若有小于1的则选最小，更新r

    # ---- 第二步：计算缩放后的实际尺寸和所需填充量 ----
    # ratio: 缩放比例元组，用于返回值
    ratio = r, r  # 高宽比，用上面计算的最小r作为宽高比
    # new_unpad: 缩放后（未填充前）的实际尺寸 (宽, 高)
    # 注意：这里是 (宽, 高) 顺序，因为后续 cv2.resize 需要 (宽, 高)
    # 例如：shape[1]*r = 1376*0.465 = 640, shape[0]*r = 776*0.465 = 361
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # round:四舍五入,宽高,注意顺序 new_unpad:(640, 361)
    # print(f'按比例需要缩放到:{new_unpad}')
    # dw, dh: 宽度和高度方向上需要填充的总像素数
    # 例如：dw = 640-640 = 0, dh = 640-361 = 279
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding (640-640) ,(640-361)
    # print(f'填充的大小,dw:{dw},dh:{dh}')

    # ---- 第三步：根据模式调��填充策略 ----
    if auto:  # 最小矩形,为False
        # auto=True 时，填充量只取 stride 的余数，实现最小填充
        # 例如：dh=279, stride=32 → dh = 279%32 = 23（只需填充 23px 而非 279px）
        # 优点：减少计算量；缺点：输出尺寸不固定
        # 本项目中 model_interface.py 传入 auto=False，不走此分支
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding   # mod:计算两数组对应位置元素的余数。
        # print(f'最小矩形dw:{dw},dh:{dh}')
    elif scaleFill:  # 缩放，一般为False
        # scaleFill=True 时，直接拉伸到目标尺寸（不填充，但会变形）
        # 通常不使用此模式
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # ---- 第四步：将填充量均分到两侧 ----
    # 例如：dh=279 → dh=139.5（上方和下方各填充约 139.5 像素）
    dw /= 2  # 将填充分为 2 条边
    dh /= 2

    # ---- 第五步：执行缩放 ----
    # shape[::-1] 将 (高,宽) 反转为 (宽,高)，与 new_unpad 格式对齐后比较
    # 只有当实际尺寸与目标不同时才进行 resize（避免无意义的计算）
    if shape[::-1] != new_unpad:  #  裁剪 shape[::-1]:(1376, 776) new_unpad:(640, 361)
        # INTER_LINEAR: 双线性插值，速度和质量平衡较好
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # ---- 第六步：添加灰色边框填充 ----
    # 由于 dh 是浮点数（如 139.5），上下两侧的填充像素数需要取整
    # 使用 ±0.1 的微调避免浮点取整导致总填充多 1 像素的问题
    # 例如：dh=139.5 → top = round(139.4) = 139, bottom = round(139.6) = 140
    # 139 + 140 = 279 = 原始 dh*2，确保总填充量精确
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))    # 防止过填充
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # copyMakeBorder: 在图像四周添加边框
    # BORDER_CONSTANT: 使用常数颜色填充（即 color 参数指定的灰色）
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # print(f'填充后的图片尺寸:{im.shape}')
    # 返回值：填充后的图像、缩放比例、单侧填充量（供 scale_coords 使用）
    return im, ratio, (dw, dh)


# =============================================================================
# 后处理函数
# =============================================================================

'''------后处理------'''
# NMS
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        max_det=300):
    """
    非极大值抑制（Non-Maximum Suppression, NMS）

    YOLOv5 模型的原始输出包含数万个候选检测框（anchors），其中绝大多数是低置信度的
    背景框，且同一个目标往往会产生多个高度重叠的检测框。NMS 的作用是：
      1. 过滤掉置信度低于阈值的候选框
      2. 对剩余的框按置信度排序
      3. 依次取最高置信度的框，去除与之 IOU 超过阈值的所有其他框
      4. 重复直到处理完所有框

    NMS 处理流程示意：

      模型原始输出                    置信度过滤后                NMS 去重后
      (25200 个候选框)               (约几十~几百个框)           (最终几个框)
      ┌─────────────┐               ┌─────────────┐           ┌─────────────┐
      │ ░░░░░░░░░░░ │               │             │           │             │
      │ ░░░░░░░░░░░ │  conf>0.25    │  ┌──┐ ┌──┐  │  IOU>0.45 │  ┌──┐       │
      │ ░░░░░░░░░░░ │  ─────────→   │  │██│ │██│  │  ────────→│  │██│       │
      │ ░░░░░░░░░░░ │               │  └──┘ └──┘  │           │  └──┘       │
      │ ░░░░░░░░░░░ │               │     ┌──┐    │           │     ┌──┐    │
      └─────────────┘               │     │██│    │           │     │██│    │
                                    │     └──┘    │           │     └──┘    │
                                    └─────────────┘           └─────────────┘

    在 model_interface.py 的 YOLOv5ONNXDetector.inference_image() 中被调用，
    参数为 conf_thres=self.confidence, iou_thres=self.iou（来自 GUI 滑块的实时值）。

    Args:
        prediction: 模型原始输出张量，形状为 (batch_size, num_anchors, 5+num_classes)
                    每个 anchor 的格式为 [cx, cy, w, h, objectness, cls1_conf, cls2_conf, ...]
                    - cx, cy, w, h: 中心点坐标和宽高（xywh 格式，模型输入尺寸空间）
                    - objectness: 目标存在的置信度（0~1）
                    - cls_conf: 各类别的条件概率（0~1），本项目有 2 个类：bolt, bulk
        conf_thres: 置信度阈值（0~1），低于此值的框被过滤。默认 0.25
                    本项目中由 GUI 滑块控制，默认 0.45
        iou_thres: IOU 阈值（0~1），NMS 中重叠度超过此值的框被抑制。默认 0.45
                   本项目中由 GUI 滑块控制，默认 0.45
        classes: 要保留的类别 ID 列表。None 表示保留所有类别。
                 例如 classes=[0] 则只保留 "bolt" 类的检测结果
        agnostic: 是否使用"类别无关"的 NMS。
                  False（默认）: 不同类别的框之间不互相抑制（通过坐标偏移实现）
                  True: 所有类别统一做 NMS（可能导致不同类别的重叠目标被误删）
        multi_label: 是否允许多标签（一个框同时属于多个类别）。
                     默认 False，本项目中锚杆(bolt)和散料(bulk)不会重叠，无需多标签
        max_det: 每张图像的最大检测框数量。默认 300，
                 本项目在 model_interface.py 中传入 max_det=1000

    Returns:
        output: 列表，长度为 batch_size，每个元素为形状 (n, 6) 的张量
                每行格式为 [x1, y1, x2, y2, confidence, class_id]
                - (x1,y1,x2,y2): xyxy 格式的边界框坐标（模型输入尺寸空间，如 640×640）
                - confidence: 最终置信度 = objectness × class_confidence
                - class_id: 类别索引，0=bolt, 1=bulk（对应 class_names.txt 的行号）
                如果某张图像没有检测结果，则对应元素为形状 (0, 6) 的空张量

    返回：检测列表，每个图像的 (n,6) tensor [xyxy, conf, cls]   # [左上角坐标xy右下角坐标xy,置信度,类别]
    """

    # ---- 第一步：计算类别数和初始候选筛选 ----
    # prediction.shape = (batch_size, num_anchors, 5+nc)
    # 例如 (1, 25200, 7)：batch=1, 25200个候选框, 5+2=7（4坐标+1目标度+2类别）
    nc = prediction.shape[2]- 5 # 类别数量
    # print(prediction.shape)
    # 根据 objectness（第5列，索引4）筛选：大于阈值的为候选框
    # xc 是布尔掩码，形状为 (batch_size, num_anchors)
    xc = prediction[..., 4] > conf_thres  # 候选框

    # ---- 第二步：设置约束参数 ----
    # 设置
    # min_wh, max_wh: 边界框的有效宽高范围（像素）
    # 小于 2px 或大于 4096px 的框被认为是无效框
    min_wh, max_wh = 2, 4096   # （像素）最小和最大盒子宽度和高度
    # max_nms: 送入 torchvision NMS 算子的最大框数（防止过多框导致计算缓慢）
    max_nms = 3000  # torchvision.ops.nms() 中的最大框数
    # multi_label: 仅在类别数 > 1 时才有意义（单类检测不需要多标签）
    multi_label &= nc > 1  # 每个候选框的多标签设置（增加 0.5ms/img）



    # ---- 第三步：初始化输出列表 ----
    # 为 batch 中每张图像创建一个空的 (0, 6) 张量作为默认输出
    # 如果某张图像没有检测结果，这个空张量就是最终输出
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    # ---- 第四步：逐张图像处理 ----
    for xi, x in enumerate(prediction):  # 图像索引xi，图像推断x    # enumerate可遍历的数据对象组合为一个索引序列
        # ---- 4.1 应用宽高约束 ----
        # 将宽或高不在 [min_wh, max_wh] 范围内的框的 objectness 置为 0（使其被过滤）
        # x[..., 2:4] 是所有框的宽高，any(1) 检查每行是否有任意一个维度超出范围
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # 宽高
        # 应用 objectness 阈值过滤：只保留 xc[xi] 为 True 的行
        x = x[xc[xi]]  # 置信度

        # 如果没有数据就处理下一个图像
        # 所有框都被过滤掉了，跳到下一张图像
        if not x.shape[0]:
            continue

        # ---- 4.2 计算最终置信度 ----
        # YOLOv5 的双重置信度机制：
        #   objectness（目标度）× class_confidence（类别条件概率）= 最终置信度
        # x[:, 4:5] 是 objectness（保持维度用于广播），x[:, 5:] 是各类别的条件概率
        # 乘法后 x[:, 5:] 中每个值就是该框属于对应类别的最终置信度
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # 将坐标从 xywh 格式转换为 xyxy 格式（NMS 需要 xyxy 格式）
        box = xywh2xyxy(x[:, :4])

        # ---- 4.3 构建检测矩阵 ----
        # 从 (N, 5+nc) 格式重组为 (M, 6) 格式：[x1, y1, x2, y2, conf, cls_id]
        if multi_label:
            # 多标签模式：同一个框可以输出多个类别的检测结果
            # 找出所有 conf > conf_thres 的 (框索引i, 类别索引j) 对
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            # 拼接为 [xyxy, conf, cls]
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # 只有最好类
            # 单标签模式（默认）：每个框只取置信度最高的类别
            # max(1): 沿类别维度取最大值，conf=最大置信度, j=对应类别索引
            conf, j = x[:, 5:].max(1, keepdim=True)
            # 拼接为 [xyxy, best_conf, best_cls]，然后再次过滤掉低置信度的框
            # 注意：之前是按 objectness 过滤，这里是按最终置信度（obj×cls）过滤
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # ---- 4.4 按指定类别过滤（可选）----
        # 如果 classes 参数不为 None，只保留指定类别的检测结果
        # 例如 classes=[0] 只保留 bolt 类
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # ---- 4.5 检查剩余框数量 ----
        n = x.shape[0]  # 框的个数
        if not n:  # 没有锚框
            # 过滤后没有剩余框，跳到下一张图像
            continue
        elif n > max_nms:  # 多余的锚框
            # 框数超过 NMS 上限时，按置信度降序排列并截取前 max_nms 个
            # 这是为了防止 NMS 算子处理过多框导致性能问题
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序

        # ---- 4.6 执行批量 NMS ----
        # "按类别 NMS" 的实现技巧：
        # 将每个框的坐标加上一个与类别相关的大偏移量 (class_id * max_wh)
        # 这样不同类别的框在坐标空间上被隔开，torchvision.ops.nms 在计算 IOU 时
        # 不同类别的框之间不会重叠，从而实现"各类别独立做 NMS"的效果
        # agnostic=True 时偏移量为 0，即所有类别统一做 NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        # boxes: 加了类别偏移的坐标；scores: 置信度
        boxes, scores = x[:, :4] + c, x[:, 4]  # 框（类别偏移），分数
        # 调用 torchvision 的 NMS 算子，返回保留框的索引
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # 限制最终检测数量不超过 max_det
        if i.shape[0] > max_det:  # 极限检测
            i = i[:max_det]

        # 将当前图像的最终检测结果写入输出列表
        output[xi] = x[i]

    return output


def clip_coords(boxes, shape):
    """
    将边界框坐标裁剪到图像范围内。

    scale_coords 将坐标从模型输入空间映射回原图空间后，由于浮点运算的精度误差，
    某些坐标可能略微超出图像边界（如 x=-0.5 或 x=641）。
    此函数将所有坐标限制在 [0, 图像宽/高] 范围内，确保不出现无效的负坐标
    或超出图像范围的坐标。

    clamp_（原地操作，PyTorch）和 clip（numpy）的作用相同：
    将值限制在 [min, max] 范围内。

    Args:
        boxes: 形状为 (N, 4) 的张量或数组，每行为 [x1, y1, x2, y2]
               支持 torch.Tensor 和 numpy.ndarray
        shape: 图像形状，(高度, 宽度) 格式（即 image.shape[:2] 的结果）
               x 坐标限制在 [0, shape[1]]，y 坐标限制在 [0, shape[0]]
    """
    # 将边界 xyxy 框裁剪为图像形状（高度、宽度）
    if isinstance(boxes, torch.Tensor):  # tensor类型
        # clamp_: 原地操作，将值限制在 [min, max] 范围内
        # shape[1] = 图像宽度，x 坐标不能超出
        boxes[:, 0].clamp_(0, shape[1])  # x1
        # shape[0] = 图像高度，y 坐标不能超出
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array 类型
        # numpy 的 clip 函数：等价于 clamp
        # 同时处理 x1 和 x2（列索引 0 和 2），限制在 [0, 宽度]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        # 同时处理 y1 和 y2（列索引 1 和 3），限制在 [0, 高度]
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


# 转换
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将检测框坐标从模型输入尺寸空间还原到原始图像尺寸空间。

    这是 letterbox 预处理的逆操作。letterbox 对图像做了两步变换：
      1. 等比例缩放（乘以 gain）
      2. 边界填充（加上 pad）
    因此还原坐标时需要反向操作：
      1. 减去填充偏移量（pad）
      2. 除以缩放比例（gain）

    坐标还原流程示意：

      模型输入空间 (640×640)           原始图像空间 (1376×776)
      ┌──────────────────┐             ┌──────────────────────┐
      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │             │                      │
      │▓▓┌──┐▓▓▓▓▓▓▓▓▓▓▓ │  -pad       │                      │
      │▓▓│██│▓▓▓▓▓▓▓▓▓▓▓ │  ÷gain      │  ┌──┐                │
      │▓▓└──┘▓▓▓▓▓▓▓▓▓▓▓ │ ─────────→  │  │██│                │
      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │             │  └──┘                │
      └──────────────────┘             └──────────────────────┘

    在 model_interface.py 的 inference_image() 中，NMS 之后调用此函数将
    检测框坐标从 640×640 空间映射回原始图像尺寸。

    Args:
        img1_shape: 模型输入图像的形状 (高, 宽)，通常为 (640, 640)
                    在调用处通过 img.shape[2:] 获取（img 是 4 维张量 [B,C,H,W]）
        coords: 形状为 (N, 4) 的坐标张量，格式 [x1, y1, x2, y2]
                这是 NMS 输出的检测框坐标，位于模型输入尺寸空间中
                此函数会原地修改此张量
        img0_shape: 原始图像的形状 (高, 宽, 通道)，即 image.shape
                    用于计算缩放比例和填充量
        ratio_pad: 可选的 (ratio, pad) 元组，直接提供缩放比例和填充量。
                   如果为 None（默认），则根据 img1_shape 和 img0_shape 重新计算

    Returns:
        coords: 还原后的坐标张量，格式不变 [x1, y1, x2, y2]，
                但坐标值已映射到原始图像的像素空间中
    """
    # 将坐标 (xyxy) 从 img_shape 重新缩放为 img0_shape
    if ratio_pad is None:  # 从 img0_shape 计算
        # 重新计算缩放比例 gain（与 letterbox 中的 r 计算方式一致）
        # gain = min(模型输入高/原图高, 模型输入宽/原图宽)
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 比例 = 旧 / 新
        # 重新计算填充偏移量 pad
        # pad[0] = (模型输入宽 - 原图宽*gain) / 2 → 水平方向单侧填充量
        # pad[1] = (模型输入高 - 原图高*gain) / 2 → 垂直方向单侧填充量
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh 填充大小
    else:
        # 使用外部提供的比例和填充量
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # ---- 逆操作1：减去填充偏移量 ----
    # 将 x 坐标（x1 和 x2）减去水平填充量
    coords[:, [0, 2]] -= pad[0]  # x 填充
    # 将 y 坐标（y1 和 y2）减去垂直填充量
    coords[:, [1, 3]] -= pad[1]  # y 填充
    # ---- 逆操作2：除以缩放比例 ----
    # 将坐标从缩放后的尺寸还原为原始尺寸
    coords[:, :4] /= gain
    # ---- 安全裁剪：确保坐标不超出原图范围 ----
    clip_coords(coords, img0_shape)
    return coords