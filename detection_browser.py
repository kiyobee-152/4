# -*- coding: utf-8 -*-
"""
检测结果浏览器模块
提供历史检测结果的查看和筛选功能
✅ 修改：刷新后保留筛选状态和列宽、支持顺序/倒序排列
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QPushButton, QLineEdit, QComboBox, 
                             QLabel, QSpinBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView
from datetime import datetime
import json
import os


class DetectionBrowser(QDialog):
    """检测结果浏览器窗口"""
    
    def __init__(self, post_processor, logger=None, parent=None):
        super().__init__(parent)
        self.post_processor = post_processor
        self.logger = logger
        
        self.setWindowTitle("检测结果浏览器")
        self.setGeometry(50, 50, 1600, 700)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
        
        # ✅ 保存筛选状态
        self.saved_feed_filter = "所有镜头"
        self.saved_class_filter = "所有类别"
        self.saved_conf_filter = 0
        self.saved_search_filter = ""
        self.saved_sort_order = "倒序"  # ✅ 新增：保存排序状态
        
        self.init_ui()
        self.load_data()
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout()
        
        # 搜索和筛选栏
        filter_layout = QHBoxLayout()
        
        # ✅ 添加"镜头"筛选器
        feed_label = QLabel("镜头:")
        self.feed_combo = QComboBox()
        self.feed_combo.addItem("所有镜头")
        for i in range(8):
            self.feed_combo.addItem(f"画面{i + 1}")
        self.feed_combo.currentIndexChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(feed_label)
        filter_layout.addWidget(self.feed_combo)
        
        class_label = QLabel("类别:")
        self.class_combo = QComboBox()
        self.class_combo.addItem("所有类别")
        self.class_combo.currentIndexChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(class_label)
        filter_layout.addWidget(self.class_combo)
        
        conf_label = QLabel("最小置信度:")
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(0, 100)
        self.conf_spin.setValue(0)
        self.conf_spin.setSuffix("%")
        self.conf_spin.valueChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(conf_label)
        filter_layout.addWidget(self.conf_spin)
        
        search_label = QLabel("搜索帧ID:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入帧ID...")
        self.search_input.textChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(search_label)
        filter_layout.addWidget(self.search_input)
        
        # ✅ 新增：排序顺序选择器
        sort_label = QLabel("排序:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("倒序")  # 最新的在前
        self.sort_combo.addItem("顺序")  # 最旧的在前
        self.sort_combo.currentIndexChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(sort_label)
        filter_layout.addWidget(self.sort_combo)
        
        filter_layout.addStretch()
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.load_data)
        export_btn = QPushButton("导出选中")
        export_btn.clicked.connect(self.export_selected)
        
        filter_layout.addWidget(refresh_btn)
        filter_layout.addWidget(export_btn)
        
        # ✅ 数据表格 - 添加"镜头"列
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "时间戳", "镜头", "帧ID", "类别", "置信度", "位置", "操作"
        ])
        # ✅ 改为 False，防止最后一列自动扩展
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        
        # ✅ 设置列宽策略：前6列自动扩展，"操作"列固定宽度
        self.table.setColumnWidth(6, 70)  # "操作"列固定70像素
        for i in range(6):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
        
        # 统计信息
        self.stats_label = QLabel()
        
        main_layout.addLayout(filter_layout)
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.stats_label)
        
        self.setLayout(main_layout)
    
    def on_filter_changed(self):
        """✅ 当筛选器改变时保存状态并重新过滤"""
        self.save_filter_state()
        self.filter_data()
    
    def save_filter_state(self):
        """✅ 保存当前筛选器状态"""
        self.saved_feed_filter = self.feed_combo.currentText()
        self.saved_class_filter = self.class_combo.currentText()
        self.saved_conf_filter = self.conf_spin.value()
        self.saved_search_filter = self.search_input.text().strip()
        self.saved_sort_order = self.sort_combo.currentText()  # ✅ 保存排序状态
    
    def restore_filter_state(self):
        """✅ 恢复之前保存的筛选器状态"""
        # 禁用信号以避免触发 on_filter_changed
        self.feed_combo.blockSignals(True)
        self.class_combo.blockSignals(True)
        self.conf_spin.blockSignals(True)
        self.search_input.blockSignals(True)
        self.sort_combo.blockSignals(True)  # ✅ 禁用排序下拉框信号
        
        # 恢复筛选器状态
        feed_idx = self.feed_combo.findText(self.saved_feed_filter)
        if feed_idx >= 0:
            self.feed_combo.setCurrentIndex(feed_idx)
        
        class_idx = self.class_combo.findText(self.saved_class_filter)
        if class_idx >= 0:
            self.class_combo.setCurrentIndex(class_idx)
        
        self.conf_spin.setValue(self.saved_conf_filter)
        self.search_input.setText(self.saved_search_filter)
        
        # ✅ 恢复排序状态
        sort_idx = self.sort_combo.findText(self.saved_sort_order)
        if sort_idx >= 0:
            self.sort_combo.setCurrentIndex(sort_idx)
        
        # 重新启用信号
        self.feed_combo.blockSignals(False)
        self.class_combo.blockSignals(False)
        self.conf_spin.blockSignals(False)
        self.search_input.blockSignals(False)
        self.sort_combo.blockSignals(False)  # ✅ 启用排序下拉框信号
    
    def load_data(self):
        """加载所有检测数据"""
        try:
            detections = self.post_processor.get_detection_history_copy()
            
            classes = set([d.class_name for d in detections])
            
            # ✅ 保留当前的类别选择
            self.class_combo.blockSignals(True)
            current_class = self.class_combo.currentText()
            self.class_combo.clear()
            self.class_combo.addItem("所有类别")
            self.class_combo.addItems(sorted(classes))
            
            # 恢复之前的选择
            if current_class != "所有类别" and self.class_combo.findText(current_class) >= 0:
                self.class_combo.setCurrentText(current_class)
            else:
                self.class_combo.setCurrentIndex(0)
            
            self.class_combo.blockSignals(False)
            
            self.all_detections = detections
            
            # ✅ 恢复筛选器状态
            self.restore_filter_state()
            
            # ✅ 应用筛选
            self.filter_data()
            
            if self.logger:
                self.logger.info(f"浏览器加载了 {len(detections)} 条检测记录")
        
        except Exception as e:
            if self.logger:
                self.logger.exception(f"加载数据失败: {e}")
            QMessageBox.warning(self, "错误", f"加载数据失败: {str(e)}")
    
    def filter_data(self):
        """根据筛选条件过滤数据"""
        try:
            # ✅ 获取镜头筛选条件
            selected_feed_text = self.feed_combo.currentText()
            selected_feed = -1 if selected_feed_text == "所有镜头" else int(selected_feed_text.replace("画面", "")) - 1
            
            selected_class = self.class_combo.currentText()
            min_confidence = self.conf_spin.value() / 100.0
            search_frame_id = self.search_input.text().strip()
            
            # ✅ 获取排序顺序
            sort_order = self.sort_combo.currentText()
            reverse = (sort_order == "倒序")  # 倒序时 reverse=True
            
            filtered = []
            for det in self.all_detections:
                # ✅ 按镜头筛选
                if selected_feed != -1 and det.feed_id != selected_feed:
                    continue
                
                if selected_class != "所有类别" and det.class_name != selected_class:
                    continue
                
                if det.confidence < min_confidence:
                    continue
                
                if search_frame_id and str(det.frame_id) != search_frame_id:
                    continue
                
                filtered.append(det)
            
            # ✅ 按排序顺序排列（默认按时间戳排序）
            filtered.sort(key=lambda d: d.timestamp, reverse=reverse)
            
            self.update_table(filtered)
            
            total = len(self.all_detections)
            shown = len(filtered)
            self.stats_label.setText(f"显示 {shown}/{total} 条记录")
        
        except Exception as e:
            if self.logger:
                self.logger.exception(f"筛选数据失败: {e}")
    
    def update_table(self, detections):
        """更新表格显示"""
        self.table.setRowCount(len(detections))
        
        # ✅ 定义强调颜色
        alert_brush = QtGui.QBrush(QtGui.QColor(255, 200, 200))  # 浅红色背景
        alert_font = QtGui.QFont()
        alert_font.setBold(True)  # 加粗字体
        
        # 需要强调的类别
        alert_classes = {'Other_garbage', 'bolt'}
        
        for row, det in enumerate(detections):
            timestamp_item = QTableWidgetItem(det.timestamp)
            self.table.setItem(row, 0, timestamp_item)
            
            # ✅ 显示镜头ID（格式化为"画面1"-"画面8"）
            feed_name = f"画面{det.feed_id + 1}"
            feed_item = QTableWidgetItem(feed_name)
            self.table.setItem(row, 1, feed_item)
            
            frame_id = str(det.frame_id) if det.frame_id is not None else "-"
            frame_item = QTableWidgetItem(frame_id)
            self.table.setItem(row, 2, frame_item)
            
            class_item = QTableWidgetItem(det.class_name)
            self.table.setItem(row, 3, class_item)
            
            conf_text = f"{det.confidence:.2f}"
            conf_item = QTableWidgetItem(conf_text)
            self.table.setItem(row, 4, conf_item)
            
            bbox_text = f"({det.bbox[0]},{det.bbox[1]},{det.bbox[2]},{det.bbox[3]})"
            bbox_item = QTableWidgetItem(bbox_text)
            self.table.setItem(row, 5, bbox_item)
            
            save_btn = QPushButton("保存")
            save_btn.clicked.connect(lambda checked, r=row: self.save_detection(r))
            self.table.setCellWidget(row, 6, save_btn)
            
            # ✅ 如果是 'Other_garbage' 或 'bolt'，强调整行
            if det.class_name in alert_classes:
                for col in range(7):
                    cell = self.table.item(row, col)
                    if cell:
                        cell.setBackground(alert_brush)
                        cell.setFont(alert_font)
                        # 设置文字颜色为深红
                        cell.setForeground(QtGui.QColor(200, 0, 0))
        
        # ✅ 修改：只调整前6列，"操作"列宽度保持70像素
        self.table.resizeColumnsToContents()
        for i in range(6):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        # 确保"操作"列保持固定宽度
        self.table.setColumnWidth(6, 70)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
    
    def export_selected(self):
        """导出选中的记录"""
        try:
            selected = self.table.selectedIndexes()
            if not selected:
                QMessageBox.warning(self, "提示", "请先选择要导出的记录")
                return
            
            rows = set([idx.row() for idx in selected])
            
            export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if not export_dir:
                return
            
            export_data = []
            for row in sorted(rows):
                timestamp = self.table.item(row, 0).text()
                
                for det in self.all_detections:
                    if det.timestamp == timestamp:
                        export_data.append(det.to_dict())
                        break
            
            export_file = os.path.join(export_dir, f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "成功", f"已导出 {len(export_data)} 条记录到:\n{export_file}")
            if self.logger:
                self.logger.info(f"导出了 {len(export_data)} 条检测记录到 {export_file}")
        
        except Exception as e:
            if self.logger:
                self.logger.exception(f"导出失败: {e}")
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def save_detection(self, row):
        """保存单条检测记录"""
        try:
            timestamp = self.table.item(row, 0).text()
            
            export_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
            if not export_dir:
                return
            
            for det in self.all_detections:
                if det.timestamp == timestamp:
                    feed_name = f"画面{det.feed_id + 1}"
                    save_file = os.path.join(export_dir, f"detection_{feed_name}_{det.class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(save_file, 'w', encoding='utf-8') as f:
                        json.dump(det.to_dict(), f, ensure_ascii=False, indent=2)
                    
                    QMessageBox.information(self, "成功", f"已保存到:\n{save_file}")
                    break
        
        except Exception as e:
            if self.logger:
                self.logger.exception(f"保存失败: {e}")
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")