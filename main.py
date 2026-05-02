# -*- coding: utf-8 -*-
"""多路(2x4)锚杆检测系统主界面 - 第二阶段改进版本"""
import os
import sys
import time
import threading
import webbrowser
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox

import image_rc
from model_interface import create_detector
from post_processor import PostProcessor
from video_processor import VideoProcessor, FrameRateController
from logger import get_logger, setup_logging
from system_monitor import get_system_metrics
from config_manager import get_config_manager
from statistics_panel import StatisticsPanel
from report_generator import ReportGenerator
from detection_browser import DetectionBrowser


class FeedPreviewLabel(QtWidgets.QLabel):
    """高辨率帧预览标签"""
    def minimumSizeHint(self):
        return QtCore.QSize(300, 150)
    def sizeHint(self):
        return QtCore.QSize(300, 150)


class Ui_MainWindow(QtWidgets.QMainWindow):
    signal = pyqtSignal(int, str, str)

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1780, 1000)
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)

        self.weights_dir = './weights'
        self.detector = None
        self.post_processor = PostProcessor()
        # 创建 8 个独立的 VideoProcessor 实例
        self.feed_video_processors = [VideoProcessor() for _ in range(8)]
        
        # 使用信号量控制推理（只允许一个线程推理）
        self.infer_semaphore = threading.Semaphore(1)

        self.logger = get_logger(__name__)
        self.config_manager = get_config_manager()
        self.report_generator = ReportGenerator(self.post_processor, self.logger)
        
        self.frame_timestamps = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)
        
        self.statistics_window = None
        self.browser_window = None
        self.feed_frame_counts = [0] * 8
        self.feed_total_frames = [0] * 8

        self.feed_panels = []
        self.feed_running_flags = [False] * 8
        self.feed_threads = [None] * 8
        self.feed_latest_frames = [None] * 8
        self.feed_current_frame_id = [0] * 8
        self.feed_last_summary = ['等待输入源'] * 8
        self.feed_last_alarm_dict = [None] * 8
        self.selected_feed_id = 0
        self._is_restoring_config = False

        self._build_left_panel()
        self._build_grid_panel()
        self._build_toolbar_menu_status()

        self.retranslateUi()
        self.init_all()

    def _build_left_panel(self):
        self.left_panel = QtWidgets.QFrame(self.centralwidget)
        self.left_panel.setGeometry(QtCore.QRect(10, 10, 300, 960))

        self.label_2 = QtWidgets.QLabel("模型选择", self.left_panel)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 120, 24))

        self.cb_weights = QtWidgets.QComboBox(self.left_panel)
        self.cb_weights.setGeometry(QtCore.QRect(10, 38, 280, 26))
        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)

        self.label_device = QtWidgets.QLabel("推理设备", self.left_panel)
        self.label_device.setGeometry(QtCore.QRect(10, 70, 120, 24))
        self.cb_device = QtWidgets.QComboBox(self.left_panel)
        self.cb_device.setGeometry(QtCore.QRect(10, 96, 280, 26))
        self.cb_device.addItems(["GPU", "CPU"])
        self.cb_device.currentIndexChanged.connect(self.device_changed)

        self.label_3 = QtWidgets.QLabel("置信度", self.left_panel)
        self.label_3.setGeometry(QtCore.QRect(10, 126, 80, 24))
        self.hs_conf = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.left_panel)
        self.hs_conf.setGeometry(QtCore.QRect(10, 152, 210, 18))
        self.hs_conf.setRange(1, 100)
        self.hs_conf.setValue(45)
        self.hs_conf.valueChanged.connect(self.conf_change)
        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.left_panel)
        self.dsb_conf.setGeometry(QtCore.QRect(230, 148, 60, 26))
        self.dsb_conf.setRange(0.01, 1.0)
        self.dsb_conf.setSingleStep(0.01)
        self.dsb_conf.setValue(0.45)
        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)

        self.label_4 = QtWidgets.QLabel("IOU", self.left_panel)
        self.label_4.setGeometry(QtCore.QRect(10, 180, 80, 24))
        self.hs_iou = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.left_panel)
        self.hs_iou.setGeometry(QtCore.QRect(10, 206, 210, 18))
        self.hs_iou.setRange(1, 100)
        self.hs_iou.setValue(45)
        self.hs_iou.valueChanged.connect(self.iou_change)
        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.left_panel)
        self.dsb_iou.setGeometry(QtCore.QRect(230, 202, 60, 26))
        self.dsb_iou.setRange(0.01, 1.0)
        self.dsb_iou.setSingleStep(0.01)
        self.dsb_iou.setValue(0.45)
        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)

        self.label_active = QtWidgets.QLabel("当前操作画面", self.left_panel)
        self.label_active.setGeometry(QtCore.QRect(10, 236, 120, 24))
        self.cb_active_feed = QtWidgets.QComboBox(self.left_panel)
        self.cb_active_feed.setGeometry(QtCore.QRect(10, 262, 280, 26))
        for i in range(8):
            self.cb_active_feed.addItem(f"画面{i + 1}")
        self.cb_active_feed.currentIndexChanged.connect(self._on_active_feed_changed)

        self.btn_start_all = QtWidgets.QPushButton("全部启动(视频/相机)", self.left_panel)
        self.btn_start_all.setGeometry(QtCore.QRect(10, 298, 135, 28))
        self.btn_start_all.clicked.connect(self.start_all_running_feeds)

        self.btn_stop_all = QtWidgets.QPushButton("全部停止", self.left_panel)
        self.btn_stop_all.setGeometry(QtCore.QRect(155, 298, 135, 28))
        self.btn_stop_all.clicked.connect(self.stop_all_feeds)

        self.label_5 = QtWidgets.QLabel("结果统计", self.left_panel)
        self.label_5.setGeometry(QtCore.QRect(10, 354, 120, 24))
        self.le_res = QtWidgets.QTextEdit(self.left_panel)
        self.le_res.setGeometry(QtCore.QRect(10, 382, 280, 276))

        self.label_record_count = QtWidgets.QLabel("检测记录: 0 条", self.left_panel)
        self.label_record_count.setGeometry(QtCore.QRect(10, 664, 280, 22))

        self.btn_statistics = QtWidgets.QPushButton("📊 数据统计", self.left_panel)
        self.btn_statistics.setGeometry(QtCore.QRect(10, 690, 135, 28))
        self.btn_statistics.clicked.connect(self.show_statistics)

        self.btn_browser = QtWidgets.QPushButton("🔍 历史记录", self.left_panel)
        self.btn_browser.setGeometry(QtCore.QRect(155, 690, 135, 28))
        self.btn_browser.clicked.connect(self.show_browser)

        self.btn_save_image = QtWidgets.QPushButton("保存当前画面", self.left_panel)
        self.btn_save_image.setGeometry(QtCore.QRect(10, 724, 135, 28))
        self.btn_save_image.clicked.connect(self.save_current_image)

        self.btn_export_csv = QtWidgets.QPushButton("导出CSV", self.left_panel)
        self.btn_export_csv.setGeometry(QtCore.QRect(155, 724, 135, 28))
        self.btn_export_csv.clicked.connect(self.export_csv)

        self.btn_export_report = QtWidgets.QPushButton("导出报告", self.left_panel)
        self.btn_export_report.setGeometry(QtCore.QRect(10, 758, 135, 28))
        self.btn_export_report.clicked.connect(self.export_html_report)

        self.btn_export_json = QtWidgets.QPushButton("导出JSON", self.left_panel)
        self.btn_export_json.setGeometry(QtCore.QRect(155, 758, 135, 28))
        self.btn_export_json.clicked.connect(self.export_json)

        self.btn_clear_history = QtWidgets.QPushButton("清空记录", self.left_panel)
        self.btn_clear_history.setGeometry(QtCore.QRect(10, 792, 280, 28))
        self.btn_clear_history.clicked.connect(self.clear_history)

    def _build_grid_panel(self):
        self.grid_scroll = QtWidgets.QScrollArea(self.centralwidget)
        self.grid_scroll.setGeometry(QtCore.QRect(320, 10, 1450, 960))
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(10)
        self.grid_layout.setVerticalSpacing(10)
        self.grid_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.grid_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # 为 8 个画面分别创建增强复选框
        self.feed_enhance_checkboxes = [None] * 8

        for idx in range(8):
            card = QtWidgets.QFrame(self.grid_container)
            card.setFrameShape(QtWidgets.QFrame.StyledPanel)
            card.setMinimumHeight(310)
            vbox = QtWidgets.QVBoxLayout(card)
            vbox.setContentsMargins(8, 8, 8, 8)

            picture = FeedPreviewLabel()
            picture.setMinimumSize(300, 150)
            picture.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            picture.setStyleSheet("background:#111827; border:1px solid #374151;")
            picture.setAlignment(QtCore.Qt.AlignCenter)
            picture.setScaledContents(True)
            picture.setText("等待输入")

            # 为每个画面添加独立的"启用图像增强"复选框
            cb_enhance = QtWidgets.QCheckBox("启用图像增强")
            cb_enhance.stateChanged.connect(lambda state, i=idx: self.toggle_feed_enhancement(i, state))
            self.feed_enhance_checkboxes[idx] = cb_enhance

            row_btn = QtWidgets.QHBoxLayout()
            btn_img = QtWidgets.QPushButton("图片")
            btn_video = QtWidgets.QPushButton("视频")
            btn_camera = QtWidgets.QPushButton("摄像头")
            btn_stop = QtWidgets.QPushButton("停止")
            btn_select = QtWidgets.QPushButton("设为当前")
            row_btn.addWidget(btn_img)
            row_btn.addWidget(btn_video)
            row_btn.addWidget(btn_camera)
            row_btn.addWidget(btn_stop)
            row_btn.addWidget(btn_select)

            btn_img.clicked.connect(lambda _, i=idx: self.open_image_for_feed(i))
            btn_video.clicked.connect(lambda _, i=idx: self.open_video_for_feed(i))
            btn_camera.clicked.connect(lambda _, i=idx: self.open_camera_for_feed(i))
            btn_stop.clicked.connect(lambda _, i=idx: self.stop_feed(i))
            btn_select.clicked.connect(lambda _, i=idx: self.set_active_feed(i))

            vbox.addWidget(picture, 1)
            vbox.addWidget(cb_enhance)
            vbox.addLayout(row_btn)

            self.feed_panels.append({
                'id': idx,
                'card': card,
                'picture': picture,
                'source_type': None,
                'source_path': None,
            })

            self.grid_layout.addWidget(card, idx // 4, idx % 4)

        for c in range(4):
            self.grid_layout.setColumnStretch(c, 1)
        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 1)

        min_row_h = 310
        min_total_h = min_row_h * 2 + self.grid_layout.verticalSpacing()
        self.grid_container.setMinimumHeight(min_total_h)

        self.grid_scroll.setWidget(self.grid_container)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 10
        left_w = 300
        top = 10
        bottom_margin = 10

        ch = self.centralwidget.height()
        total_h = max(300, ch - top - bottom_margin)
        self.left_panel.setGeometry(QtCore.QRect(margin, top, left_w, total_h))

        grid_x = margin + left_w + 10
        grid_w = max(400, self.width() - grid_x - margin)
        self.grid_scroll.setGeometry(QtCore.QRect(grid_x, top, grid_w, total_h))

        top_controls_h = 338
        bottom_fixed_h = 220
        available = max(120, total_h - top_controls_h - bottom_fixed_h)
        self.le_res.setGeometry(QtCore.QRect(10, 338, 280, available))

        y = 338 + available + 8
        self.label_record_count.setGeometry(QtCore.QRect(10, y, 280, 22))
        y += 28
        self.btn_statistics.setGeometry(QtCore.QRect(10, y, 135, 28))
        self.btn_browser.setGeometry(QtCore.QRect(155, y, 135, 28))
        y += 36
        self.btn_save_image.setGeometry(QtCore.QRect(10, y, 135, 28))
        self.btn_export_csv.setGeometry(QtCore.QRect(155, y, 135, 28))
        y += 36
        self.btn_export_report.setGeometry(QtCore.QRect(10, y, 135, 28))
        self.btn_export_json.setGeometry(QtCore.QRect(155, y, 135, 28))
        y += 36
        self.btn_clear_history.setGeometry(QtCore.QRect(10, y, 280, 28))

    def _build_toolbar_menu_status(self):
        self.menubar = QtWidgets.QMenuBar(self)
        self.setMenuBar(self.menubar)
        self.menu_file = QtWidgets.QMenu("文件", self.menubar)
        self.menubar.addAction(self.menu_file.menuAction())

        self.action_save_result = QtWidgets.QAction("保存检测结果", self)
        self.action_save_result.triggered.connect(self.save_current_image)
        self.menu_file.addAction(self.action_save_result)

        self.action_export_json = QtWidgets.QAction("导出JSON", self)
        self.action_export_json.triggered.connect(self.export_json)
        self.menu_file.addAction(self.action_export_json)

        self.menu_analysis = QtWidgets.QMenu("数据分析", self.menubar)
        self.menubar.addAction(self.menu_analysis.menuAction())

        self.action_statistics = QtWidgets.QAction("📊 数据统计", self)
        self.action_statistics.triggered.connect(self.show_statistics)
        self.menu_analysis.addAction(self.action_statistics)

        self.action_html_report = QtWidgets.QAction("📄 HTML报告", self)
        self.action_html_report.triggered.connect(self.export_html_report)
        self.menu_analysis.addAction(self.action_html_report)

        self.action_browser = QtWidgets.QAction("🔍 历史记录", self)
        self.action_browser.triggered.connect(self.show_browser)
        self.menu_analysis.addAction(self.action_browser)

        self.toolBar = QtWidgets.QToolBar(self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.actionopenpic = QtWidgets.QAction(QtGui.QIcon(":/images/1.png"), "当前画面选图片", self)
        self.actionopenpic.triggered.connect(lambda: self.open_image_for_feed(self.selected_feed_id))
        self.toolBar.addAction(self.actionopenpic)

        self.action = QtWidgets.QAction(QtGui.QIcon(":/images/2.png"), "当前画面选视频", self)
        self.action.triggered.connect(lambda: self.open_video_for_feed(self.selected_feed_id))
        self.toolBar.addAction(self.action)

        self.action_2 = QtWidgets.QAction(QtGui.QIcon(":/images/3.png"), "当前画面开摄像头", self)
        self.action_2.triggered.connect(lambda: self.open_camera_for_feed(self.selected_feed_id))
        self.toolBar.addAction(self.action_2)

        self.actionexit = QtWidgets.QAction(QtGui.QIcon(":/images/4.png"), "退出程序", self)
        self.actionexit.triggered.connect(self.exit)
        self.toolBar.addAction(self.actionexit)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.statusLabel = QLabel("状态说明   ")
        self.ssl_show = QLabel("等待操作中...")
        self.ssl_show.setFixedSize(1000, 32)
        self.statusbar.addWidget(self.statusLabel)
        self.statusbar.addWidget(self.ssl_show)

        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setFixedWidth(200)
        self.lbl_infer_time = QLabel("推理: 0ms")
        self.lbl_infer_time.setFixedWidth(200)
        self.lbl_cpu = QLabel("CPU: 0%")
        self.lbl_cpu.setFixedWidth(200)
        self.lbl_mem = QLabel("内存: 0%")
        self.lbl_mem.setFixedWidth(200)
        self.lbl_gpu = QLabel("GPU: 0%")
        self.lbl_gpu.setFixedWidth(200)

        self.statusbar.addWidget(self.lbl_fps)
        self.statusbar.addWidget(self.lbl_infer_time)
        self.statusbar.addWidget(self.lbl_cpu)
        self.statusbar.addWidget(self.lbl_mem)
        self.statusbar.addWidget(self.lbl_gpu)

        self.timer_metrics = QtCore.QTimer(self)
        self.timer_metrics.timeout.connect(self._update_status_metrics)
        self.timer_metrics.start(2000)

    def init_all(self):
        self.signal.connect(self.set_res, QtCore.Qt.QueuedConnection)
        self.load_weights_to_list()
        self._restore_config()
        if self.cb_weights.count() > 0 and self.detector is None:
            self.cb_weights_changed(show_fallback_dialog=False)
        self.beautify_left_panel()
        self.update_record_count()

    def set_active_feed(self, feed_id):
        self.cb_active_feed.setCurrentIndex(feed_id)

    def _on_active_feed_changed(self, idx):
        self.selected_feed_id = idx
        self.refresh_result_for_feed(idx)

    def cb_weights_changed(self, show_fallback_dialog=True):
        if self.cb_weights.currentText() == "":
            return
        
        title = self.windowTitle()
        self.setWindowTitle('正在加载模型中..')
        try:
            weights_path = os.path.join(self.weights_dir, self.cb_weights.currentText())
            device_preference = self.cb_device.currentText().lower()
            self.detector = create_detector(weights_path, model_type='auto',
                                            conf_thres=self.dsb_conf.value(),
                                            iou_thres=self.dsb_iou.value(),
                                            device_preference=device_preference)
            self.setWindowTitle(title)
            actual_device = "GPU" if getattr(self.detector, "device", "cpu") == "cuda" else "CPU"
            self.ssl_show.setText(f"模型加载成功: {self.cb_weights.currentText()} ({actual_device})")
            self.logger.info(f"模型加载成功: {self.cb_weights.currentText()}, 请求设备={device_preference.upper()}, 实际设备={actual_device}")
            if show_fallback_dialog and device_preference == 'gpu' and actual_device != 'GPU':
                QMessageBox.information(self, "提示", "当前环境不可用 GPU，已自动回退为 CPU。")
            self._save_config()
        except Exception as e:
            self.setWindowTitle(title)
            self.logger.exception(f"模型加载失败: {e}")
            QMessageBox.warning(self, "错误", f"模型加载失败: {str(e)}")

    def open_image_for_feed(self, feed_id):
        imgName, _ = QFileDialog.getOpenFileName(self, f"画面{feed_id+1}打开图片", "",
                                                 "图片文件 (*.jpg *.jpeg *.bmp *.png);;All Files(*)")
        if imgName:
            self.feed_panels[feed_id]['source_type'] = 'image'
            self.feed_panels[feed_id]['source_path'] = imgName
            self.stop_feed(feed_id)
            threading.Thread(target=self.start_image, args=(feed_id, imgName), daemon=True).start()

    def open_video_for_feed(self, feed_id):
        fileName, _ = QFileDialog.getOpenFileName(self, f"画面{feed_id+1}选视频", os.getcwd(),
                                                  "Video File(*.mp4 *.avi *.flv)")
        if fileName:
            self.feed_panels[feed_id]['source_type'] = 'video'
            self.feed_panels[feed_id]['source_path'] = fileName
            self.start_feed(feed_id, 'video', fileName)

    def open_camera_for_feed(self, feed_id):
        self.feed_panels[feed_id]['source_type'] = 'camera'
        self.feed_panels[feed_id]['source_path'] = 0
        self.start_feed(feed_id, 'camera', 0)

    def start_feed(self, feed_id, source_type, source_path):
        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先选择模型!")
            return
        self.stop_feed(feed_id)
        self.feed_running_flags[feed_id] = True
        if source_type == 'video':
            th = threading.Thread(target=self.start_video, args=(feed_id, source_path), daemon=True)
        else:
            th = threading.Thread(target=self.start_camera, args=(feed_id, source_path), daemon=True)
        self.feed_threads[feed_id] = th
        th.start()

    def start_camera(self, feed_id, camera_index=0):
        cap = None
        frame_rate_controller = FrameRateController(target_fps=30)
        try:
            self.signal.emit(feed_id, '正在检测摄像头中...', 'status')
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_index}")
            
            self.feed_current_frame_id[feed_id] = 0
            
            while self.feed_running_flags[feed_id]:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    self._process_and_render_frame(feed_id, frame)
                except Exception as e:
                    self.logger.warning(f"处理帧失败 (画面{feed_id+1}): {e}")
                    continue
                self.feed_current_frame_id[feed_id] += 1
                frame_rate_controller.wait_if_needed()
            
            self.logger.info(f"摄像头检测已停止 (画面{feed_id+1})")
        except Exception as e:
            self.logger.exception(f"摄像头错误 (画面{feed_id+1}): {e}")
            self.signal.emit(feed_id, f"摄像头错误: {str(e)}", 'status')
        finally:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
            self.feed_running_flags[feed_id] = False

    def start_video(self, feed_id, video_file):
        cap = None
        frame_rate_controller = FrameRateController(target_fps=30)
        try:
            self.signal.emit(feed_id, '正在检测视频中...', 'status')
            # 使用 feed_id 对应的 VideoProcessor 获取视频信息
            info = self.feed_video_processors[feed_id].get_video_info(video_file)
            if info:
                frame_rate_controller.set_fps(info.get('fps', 30))
            
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {video_file}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.feed_total_frames[feed_id] = total_frames
            self.feed_frame_counts[feed_id] = 0
            
            self.feed_current_frame_id[feed_id] = 0
            while self.feed_running_flags[feed_id]:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    self._process_and_render_frame(feed_id, frame)
                except Exception as e:
                    self.logger.warning(f"处理帧失败 (画面{feed_id+1}): {e}")
                    continue
                
                self.feed_current_frame_id[feed_id] += 1
                self.feed_frame_counts[feed_id] += 1
                
                fps = info.get('fps', 30) if info else 30
                remaining_frames = total_frames - self.feed_frame_counts[feed_id]
                remaining_seconds = remaining_frames / fps if fps > 0 else 0
                
                self.signal.emit(feed_id, f"处理中 | 剩余: {int(remaining_seconds)}秒", 'progress')
                
                frame_rate_controller.wait_if_needed()
            
            self.logger.info(f"视频检测已停止 (画面{feed_id+1}): {video_file}")
        except Exception as e:
            self.logger.exception(f"视频错误 (画面{feed_id+1}): {e}")
            self.signal.emit(feed_id, f"视频错误: {str(e)}", 'status')
        finally:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
            self.feed_running_flags[feed_id] = False

    def start_image(self, feed_id, image_path):
        try:
            if self.detector is None:
                raise RuntimeError("未加载模型")
            
            frame = cv2.imread(image_path)
            if frame is None:
                raise RuntimeError(f"无法读取图像: {image_path}")
            
            self._process_and_render_frame(feed_id, frame)
            self.logger.info(f"图片检测已完成 (画面{feed_id+1}): {image_path}")
        except Exception as e:
            self.logger.exception(f"图像错误 (画面{feed_id+1}): {e}")
            self.signal.emit(feed_id, f"图像错误: {str(e)}", 'status')

    def _process_and_render_frame(self, feed_id, frame):
        """处理并渲染帧 - 使用信号量控制推理"""
        try:
            # 使用 feed_id 对应的 VideoProcessor 处理帧
            frame = self.feed_video_processors[feed_id].process_frame(frame)
            
            infer_start = time.perf_counter()
            
            # 使用信号量，自动排队推理（无超时）
            with self.infer_semaphore:
                result_lists = self.detector.inference_image(frame)
                frame = self.detector.draw_image(result_lists, frame)
            
            infer_time = (time.perf_counter() - infer_start) * 1000
            
            self.inference_times.append(infer_time)
            self.frame_timestamps.append(time.time())
            
            # ✅ 修改：只统计需要报警的类别（large_sized_coal 不需要报警）
            if result_lists:
                rd = {}
                for result in result_lists:
                    name = result[0]
                    # large_sized_coal 是正常物体，不计入报警
                    if name != 'large_sized_coal':
                        rd[name] = rd.get(name, 0) + 1
                self.feed_last_alarm_dict[feed_id] = rd if rd else None
            else:
                self.feed_last_alarm_dict[feed_id] = None
            
            # 传入 feed_id
            self.post_processor.add_detection(result_lists, frame_id=self.feed_current_frame_id[feed_id],
                                             feed_id=feed_id)
            self.feed_latest_frames[feed_id] = frame.copy()
            summary = self.get_result_str(feed_id, result_lists)
            self.feed_last_summary[feed_id] = summary
            self.signal.emit(feed_id, summary, 'res')
            self.signal.emit(feed_id, '', 'frame_done')
        except Exception as e:
            self.logger.exception(f"帧处理错误 (画面{feed_id+1}): {e}")
            self.signal.emit(feed_id, f"处理错误: {str(e)}", 'status')

    def get_result_str(self, feed_id, result_list):
        result_dict = {}
        for result in result_list:
            name = result[0]
            result_dict[name] = result_dict.get(name, 0) + 1
        stats = self.post_processor.get_statistics()

        res = f'画面{feed_id + 1} 当前帧检测结果:\n' + '-' * 24 + '\n'
        if result_dict:
            for k, v in result_dict.items():
                res += f"{k}: {v}\n"
        else:
            res += "无目标\n"

        if stats:
            res += '\n累计统计:\n' + '-' * 24 + '\n'
            total = sum(stats.values())
            for k, v in stats.items():
                p = (v / total * 100) if total > 0 else 0
                res += f"{k}: {v} ({p:.1f}%)\n"
        return res

    def refresh_result_for_feed(self, feed_id):
        self.le_res.setPlainText(self.feed_last_summary[feed_id])

    def set_res(self, feed_id, text, flag):
        if flag == 'res':
            if feed_id == self.selected_feed_id:
                self.le_res.setPlainText(text)
            self.update_record_count()
        elif flag == 'status':
            self.ssl_show.setText(f"画面{feed_id+1}: {text}")
        elif flag == 'progress':
            self.ssl_show.setText(f"画面{feed_id+1}: {text}")
        elif flag == 'frame_done':
            panel = self.feed_panels[feed_id]
            frame = self.feed_latest_frames[feed_id]
            if frame is not None:
                lbl = panel['picture']
                tw, th = max(2, lbl.width()), max(2, lbl.height())
                display_bgr = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                rd = self.feed_last_alarm_dict[feed_id]
                if rd:
                    self._draw_alarm_banner_bgr(display_bgr, rd)
                frame_rgb = np.ascontiguousarray(
                    cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB))
                h, w = frame_rgb.shape[:2]
                img = QImage(
                    frame_rgb.data, w, h, frame_rgb.strides[0],
                    QImage.Format_RGB888).copy()
                panel['picture'].setPixmap(QPixmap.fromImage(img))

    def _draw_alarm_banner_bgr(self, frame_bgr, result_dict):
        """
        绘制检测结果报警横幅
        ✅ 修改：自动过滤 'large_sized_coal'，不显示报警
        """
        if not result_dict:
            return frame_bgr
        
        # ✅ 过滤不需要报警的类别
        alert_dict = {k: v for k, v in result_dict.items() if k != 'large_sized_coal'}
        
        # 如果没有需要报警的类别，直接返回（不绘制横幅）
        if not alert_dict:
            return frame_bgr
        
        h, w = frame_bgr.shape[:2]
        bar_h = max(18, int(round(h * 0.088)))
        font_scale = max(0.28, min(1.55, h * 0.00115))
        thickness = max(1, min(3, int(round(h / 230))))
        margin_x = max(6, int(round(w * 0.015)))
        margin_right = max(8, int(round(w * 0.012)))
        max_text_w = max(20, w - margin_x - margin_right)
        text_y = max(int(font_scale * 15 + 4), min(h - 1, int(bar_h * 0.72)))

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        summary = ", ".join([f"{k}:{v}" for k, v in alert_dict.items()])
        alarm_text = f"{timestamp} ALERT: {summary}"[:200]
        face = cv2.FONT_HERSHEY_SIMPLEX
        while len(alarm_text) > 1:
            tw, _ = cv2.getTextSize(alarm_text, face, font_scale, thickness)[0]
            if tw <= max_text_w:
                break
            alarm_text = alarm_text[:-1].rstrip()

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)
        cv2.putText(frame_bgr, alarm_text, (margin_x, text_y), face,
                    font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        return frame_bgr

    def start_all_running_feeds(self):
        restarted = 0
        for i, panel in enumerate(self.feed_panels):
            st, sp = panel['source_type'], panel['source_path']
            if st == 'video' and sp:
                self.start_feed(i, 'video', sp)
                restarted += 1
            elif st == 'camera':
                self.start_feed(i, 'camera', sp if sp is not None else 0)
                restarted += 1
        self.ssl_show.setText(f"批量启动完成: {restarted} 路")

    def stop_feed(self, feed_id):
        self.feed_running_flags[feed_id] = False
        self.ssl_show.setText(f"画面{feed_id+1}: 已停止")

    def stop_all_feeds(self):
        for i in range(8):
            self.stop_feed(i)
        self.ssl_show.setText("全部画面已停止")

    def update_record_count(self):
        count = self.post_processor.get_history_count()
        self.label_record_count.setText(f"检测记录: {count} 条")

    def save_current_image(self):
        feed_id = self.selected_feed_id
        current_frame = self.feed_latest_frames[feed_id]
        if current_frame is None:
            QMessageBox.warning(self, "警告", "当前画面没有可保存图像!")
            return
        default_dir = os.path.join(self.post_processor.output_dir, 'images')
        os.makedirs(default_dir, exist_ok=True)
        default_filename = os.path.join(default_dir, f"feed{feed_id+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        filename, _ = QFileDialog.getSaveFileName(self, "保存图像", default_filename,
                                                  "JPEG图像 (*.jpg *.jpeg);;PNG图像 (*.png);;BMP图像 (*.bmp)")
        if filename:
            to_save = current_frame.copy()
            rd = self.feed_last_alarm_dict[feed_id]
            if rd:
                self._draw_alarm_banner_bgr(to_save, rd)
            cv2.imwrite(filename, to_save)
            QMessageBox.information(self, "成功", f"图像已保存到:\n{filename}")

    def export_report(self):
        if self.post_processor.get_history_count() == 0:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!")
            return
        default_filename = os.path.join(self.post_processor.output_dir,
                                        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        filename, _ = QFileDialog.getSaveFileName(self, "导出报告", default_filename, "文本文件 (*.txt)")
        if filename:
            filepath = self.post_processor.export_report(filename)
            QMessageBox.information(self, "成功", f"报告已导出到:\n{filepath}")

    def export_html_report(self):
        if self.post_processor.get_history_count() == 0:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!")
            return
        
        try:
            report_file = self.report_generator.generate_html_report()
            if report_file:
                webbrowser.open('file://' + os.path.abspath(report_file))
                QMessageBox.information(self, "成功", f"HTML报告已生成并打开:\n{report_file}")
                self.logger.info(f"HTML报告已生成: {report_file}")
        except Exception as e:
            self.logger.exception(f"生成HTML报告失败: {e}")
            QMessageBox.critical(self, "错误", f"生成HTML报告失败: {str(e)}")

    def export_csv(self):
        if self.post_processor.get_history_count() == 0:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!")
            return
        default_filename = os.path.join(self.post_processor.output_dir,
                                        f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        filename, _ = QFileDialog.getSaveFileName(self, "导出CSV", default_filename, "CSV文件 (*.csv)")
        if filename:
            filepath = self.post_processor.export_csv(filename)
            QMessageBox.information(self, "成功", f"CSV已导出到:\n{filepath}")

    def export_json(self):
        if self.post_processor.get_history_count() == 0:
            QMessageBox.warning(self, "警告", "没有检测记录可导出!")
            return
        default_filename = os.path.join(self.post_processor.output_dir,
                                        f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        filename, _ = QFileDialog.getSaveFileName(self, "导出JSON", default_filename, "JSON文件 (*.json)")
        if filename:
            filepath = self.post_processor.export_json(filename)
            QMessageBox.information(self, "成功", f"JSON已导出到:\n{filepath}")

    def clear_history(self):
        self.post_processor.clear_history()
        self.le_res.clear()
        self.update_record_count()
        QMessageBox.information(self, "成功", "检测记录已清空!")

    def toggle_feed_enhancement(self, feed_id, state):
        """为特定 feed_id 的 VideoProcessor 切换图像增强"""
        on = state == QtCore.Qt.Checked
        vp = self.feed_video_processors[feed_id]
        vp.enable_enhancement = on
        if on:
            vp.set_enhancement_params(
                brightness=1.42, contrast=1.18, saturation=1.08)
        else:
            vp.set_enhancement_params(
                brightness=1.0, contrast=1.0, saturation=1.0)
        self._save_config()

    def load_weights_to_list(self):
        try:
            if not os.path.exists(self.weights_dir):
                self.logger.warning(f"权重目录不存在: {self.weights_dir}")
                return
            
            for file in os.listdir(self.weights_dir):
                if file.endswith('.onnx') or file.endswith('.pt'):
                    self.cb_weights.addItem(file)
            
            self.logger.info(f"加载了 {self.cb_weights.count()} 个模型文件")
        except Exception as e:
            self.logger.exception(f"加载权重文件失败: {e}")

    def iou_change(self):
        self.dsb_iou.setValue(self.hs_iou.value() / 100)
        self._save_config()

    def conf_change(self):
        self.dsb_conf.setValue(self.hs_conf.value() / 100)
        self._save_config()

    def dsb_iou_change(self):
        self.hs_iou.setValue(int(self.dsb_iou.value() * 100))
        if self.detector is not None:
            self.detector.set_iou(self.dsb_iou.value())
        self._save_config()

    def dsb_conf_change(self):
        self.hs_conf.setValue(int(self.dsb_conf.value() * 100))
        if self.detector is not None:
            self.detector.set_confidence(self.dsb_conf.value())
        self._save_config()

    def exit(self):
        self.logger.info("程序正在退出...")
        self._save_config()
        self.stop_all_feeds()
        app = QApplication.instance()
        app.quit()

    def beautify_left_panel(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f3f4f6; }
            QFrame { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
            QLabel { color: #111827; font-size: 10pt; }
            QPushButton { background: #eef2ff; border: 1px solid #c7d2fe; padding: 4px; border-radius: 6px; }
            QPushButton:hover { background: #e0e7ff; }
            QTextEdit, QComboBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; }
            QStatusBar, QToolBar { background: #eef2ff; }
        """)

    def _restore_config(self):
        try:
            self._is_restoring_config = True
            conf = self.config_manager.get('confidence', 0.45)
            iou = self.config_manager.get('iou', 0.45)
            weights = self.config_manager.get('weights', '')
            device = self.config_manager.get('device', 'gpu')
            # 恢复每个 feed 的增强设置，支持向后兼容
            feed_enhancements = self.config_manager.get('feed_enhancements', [False] * 8)
            if not isinstance(feed_enhancements, list):
                feed_enhancements = [False] * 8
            feed_enhancements = (list(feed_enhancements) + [False] * 8)[:8]
            
            self.dsb_conf.setValue(conf)
            self.dsb_iou.setValue(iou)
            if str(device).upper() in ['CPU', 'GPU']:
                self.cb_device.blockSignals(True)
                self.cb_device.setCurrentText(str(device).upper())
                self.cb_device.blockSignals(False)
            for i, checked in enumerate(feed_enhancements):
                self.feed_enhance_checkboxes[i].setChecked(bool(checked))
            
            if weights and weights in [self.cb_weights.itemText(i) for i in range(self.cb_weights.count())]:
                self.cb_weights.blockSignals(True)
                self.cb_weights.setCurrentText(weights)
                self.cb_weights.blockSignals(False)
                self.cb_weights_changed(show_fallback_dialog=False)
            
            self.logger.info("参数恢复成功")
        except Exception as e:
            self.logger.warning(f"参数恢复失败: {e}")
        finally:
            self._is_restoring_config = False
    
    def _save_config(self):
        try:
            # 保存每个 feed 的增强设置
            self.config_manager.update({
                'confidence': self.dsb_conf.value(),
                'iou': self.dsb_iou.value(),
                'weights': self.cb_weights.currentText(),
                'device': self.cb_device.currentText().lower(),
                'feed_enhancements': [cb.isChecked() for cb in self.feed_enhance_checkboxes]
            })
        except Exception as e:
            self.logger.warning(f"参数保存失败: {e}")

    def device_changed(self, _=None):
        if self._is_restoring_config or self.cb_weights.currentText() == "":
            return
        self.cb_weights_changed()
    
    def _update_status_metrics(self):
        try:
            metrics = get_system_metrics()
            self.lbl_cpu.setText(f"CPU: {metrics['cpu_percent']:.1f}%")
            self.lbl_mem.setText(f"内存: {metrics['memory_percent']:.1f}%")
            self.lbl_gpu.setText(f"GPU: {metrics['gpu_percent']:.1f}%")
            
            if self.frame_timestamps and len(self.frame_timestamps) > 1:
                fps = (len(self.frame_timestamps) - 1) / (self.frame_timestamps[-1] - self.frame_timestamps[0] + 0.001)
                self.lbl_fps.setText(f"FPS: {fps:.1f}")
            
            if self.inference_times:
                avg_time = sum(self.inference_times) / len(self.inference_times)
                self.lbl_infer_time.setText(f"推理: {avg_time:.1f}ms")
        except Exception as e:
            self.logger.debug(f"指标更新失败: {e}")

    def show_statistics(self):
        """显示数据统计窗口"""
        try:
            if self.statistics_window is None:
                self.statistics_window = StatisticsPanel(self.post_processor, self)
            self.statistics_window.update_charts()
            self.statistics_window.show()
            self.statistics_window.raise_()
            self.statistics_window.activateWindow()
        except Exception as e:
            self.logger.exception(f"打开统计窗口失败: {e}")
            QMessageBox.critical(self, "错误", f"打开统计窗口失败: {str(e)}")
    
    def show_browser(self):
        """显示检测结果浏览器"""
        try:
            if self.browser_window is None or not self.browser_window.isVisible():
                self.browser_window = DetectionBrowser(self.post_processor, self.logger, self)
                self.browser_window.show()
            else:
                self.browser_window.load_data()
                self.browser_window.raise_()
                self.browser_window.activateWindow()
        except Exception as e:
            self.logger.exception(f"打开浏览器失败: {e}")
            QMessageBox.critical(self, "错误", f"打开浏览器失败: {str(e)}")

    def retranslateUi(self):
        self.setWindowTitle("基于深度学习的皮带传送带锚杆检测系统 - v2.0")


if __name__ == "__main__":
    setup_logging()
    logger = get_logger(__name__)
    logger.info("应用程序启动")
    
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    ui.show()
    sys.exit(app.exec_())