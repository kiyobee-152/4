# -*- coding: utf-8 -*-
"""数据统计图表模块 - 支持8个镜头统一统计"""
import sys
import warnings
warnings.filterwarnings('ignore')

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QMessageBox, QTabWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class StatisticsPanel(QDialog):
    def __init__(self, post_processor, parent=None):
        super().__init__(parent)
        self.post_processor = post_processor
        self.setWindowTitle("检测数据统计分析")
        self.setGeometry(100, 100, 1400, 800)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
        
        self.figure = None
        self.canvas = None
        self.feed_combo = None
        
        self.init_ui()
        QTimer.singleShot(100, self.update_charts)
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 控制栏
        control_layout = QHBoxLayout()
        
        # ✅ 镜头选择
        feed_label = QLabel("数据来源:")
        self.feed_combo = QComboBox()
        self.feed_combo.addItem("所有镜头 (统一统计)")
        for i in range(8):
            self.feed_combo.addItem(f"画面{i + 1}")
        self.feed_combo.currentIndexChanged.connect(self.update_charts)
        
        # 图表类型选择
        chart_label = QLabel("图表类型:")
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(["柱状图", "饼图", "两个都显示"])
        self.chart_combo.currentIndexChanged.connect(self.update_charts)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.update_charts)
        refresh_btn.setFixedWidth(80)
        
        export_btn = QPushButton("导出图表")
        export_btn.clicked.connect(self.export_charts)
        export_btn.setFixedWidth(80)
        
        control_layout.addWidget(feed_label)
        control_layout.addWidget(self.feed_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(chart_label)
        control_layout.addWidget(self.chart_combo)
        control_layout.addStretch()
        control_layout.addWidget(refresh_btn)
        control_layout.addWidget(export_btn)
        
        main_layout.addLayout(control_layout)
        
        # 创建图表
        self.figure = Figure(figsize=(14, 7), dpi=100)
        self.figure.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(600)
        
        main_layout.addWidget(self.canvas, 1)
        self.setLayout(main_layout)
    
    def get_filtered_stats(self):
        """获取筛选后的统计数据"""
        detections = self.post_processor.get_detection_history_copy()
        
        # 获取选中的镜头
        feed_text = self.feed_combo.currentText()
        
        if feed_text == "所有镜头 (统一统计)":
            # 合并所有镜头的统计
            filtered_dets = detections
        else:
            # 筛选特定镜头
            feed_id = int(feed_text.split()[0][2]) - 1  # 从"画面1"提取0
            filtered_dets = [d for d in detections if d.feed_id == feed_id]
        
        # 统计数据
        stats = {}
        for det in filtered_dets:
            stats[det.class_name] = stats.get(det.class_name, 0) + 1
        
        return stats, filtered_dets
    
    def update_charts(self):
        """更新图表数据"""
        try:
            self.figure.clear()
            
            stats, detections = self.get_filtered_stats()
            
            if not stats or len(stats) == 0:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "暂无检测数据", ha='center', va='center', 
                       fontsize=16, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                self.canvas.draw()
                return
            
            chart_type = self.chart_combo.currentText()
            
            if chart_type == "柱状图":
                self._draw_bar_chart(stats, detections)
            elif chart_type == "饼图":
                self._draw_pie_chart(stats, detections)
            else:
                self._draw_both_charts(stats, detections)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"[Statistics] 错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_bar_chart(self, stats, detections):
        """绘制柱状图"""
        ax = self.figure.add_subplot(111)
        
        classes = list(stats.keys())
        counts = list(stats.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        colors = colors * (len(classes) // len(colors) + 1)
        colors = colors[:len(classes)]
        
        bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('检测类别', fontsize=13, fontweight='bold')
        ax.set_ylabel('检测数量', fontsize=13, fontweight='bold')
        ax.set_title(f'各类别检测数量统计 (总计: {len(detections)}条)', fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
    
    def _draw_pie_chart(self, stats, detections):
        """绘制饼图"""
        ax = self.figure.add_subplot(111)
        
        classes = list(stats.keys())
        counts = list(stats.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        colors = colors * (len(classes) // len(colors) + 1)
        colors = colors[:len(classes)]
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=classes, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        ax.set_title(f'各类别检测比例分布 (总计: {len(detections)}条)', fontsize=15, fontweight='bold', pad=20)
    
    def _draw_both_charts(self, stats, detections):
        """同时绘制柱状图和饼图"""
        classes = list(stats.keys())
        counts = list(stats.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        colors = colors * (len(classes) // len(colors) + 1)
        colors = colors[:len(classes)]
        
        # 左侧柱状图
        ax1 = self.figure.add_subplot(121)
        bars = ax1.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('检测类别', fontsize=12, fontweight='bold')
        ax1.set_ylabel('检测数量', fontsize=12, fontweight='bold')
        ax1.set_title(f'检测数量 (总计: {len(detections)}条)', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        
        # 右侧饼图
        ax2 = self.figure.add_subplot(122)
        wedges, texts, autotexts = ax2.pie(
            counts,
            labels=classes,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        ax2.set_title(f'检测比例', fontsize=13, fontweight='bold')
    
    def export_charts(self):
        """导出图表为图片"""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出图表", "", "PNG图像 (*.png);;JPG图像 (*.jpg);;PDF文档 (*.pdf)"
        )
        
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "成功", f"图表已导出到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")