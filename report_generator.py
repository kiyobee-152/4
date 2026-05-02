# -*- coding: utf-8 -*-
"""
HTML报告生成模块 - 功能丰富的检测报告
支持8个镜头的统一统计、镜头对比、分块分页、实时筛选等
"""
import os
from datetime import datetime
from collections import defaultdict


class ReportGenerator:
    """HTML报告生成器"""
    
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>锚杆检测报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 40px 20px;
            margin-bottom: 30px;
        }}
        
        header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .report-meta {{
            font-size: 14px;
            opacity: 0.9;
            margin-top: 15px;
        }}
        
        .report-meta p {{
            margin: 8px 0;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        section {{
            margin-bottom: 50px;
        }}
        
        section h2 {{
            font-size: 24px;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            text-align: center;
        }}
        
        .summary-card .label {{
            font-size: 12px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .progress-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
            height: 20px;
        }}
        
        .progress {{
            flex: 1;
            height: 100%;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }}
        
        .feed-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .feed-card {{
            padding: 20px;
            background-color: #f9f9f9;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}
        
        .feed-card h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .feed-stats-item {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 14px;
        }}
        
        .stat-value {{
            font-weight: bold;
            color: #764ba2;
        }}
        
        /* ✅ 新增：详细记录筛选和分页样式 */
        .filter-section {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .filter-group label {{
            font-weight: 600;
            color: #667eea;
            font-size: 13px;
        }}
        
        .filter-group select {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        
        .pagination button {{
            padding: 8px 15px;
            border: 1px solid #ddd;
            background-color: white;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        .pagination button:hover {{
            background-color: #667eea;
            color: white;
        }}
        
        .pagination button.active {{
            background-color: #667eea;
            color: white;
            border-color: #667eea;
        }}
        
        .pagination button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .page-info {{
            text-align: center;
            color: #666;
            font-size: 14px;
        }}
        
        .alert-row {{
            background-color: #ffe0e0 !important;
        }}
        
        .alert-row td {{
            color: #c00;
            font-weight: bold;
        }}
        
        .tag {{
            display: inline-block;
            background-color: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-right: 8px;
        }}
        
        .hidden {{
            display: none !important;
        }}
        
        footer {{
            text-align: center;
            border-top: 1px solid #e0e0e0;
            padding: 20px;
            color: #999;
            font-size: 12px;
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 皮带传送带锚杆检测报告</h1>
            <div class="report-meta">
                <p><strong>生成时间:</strong> {generate_time}</p>
                <p><strong>检测时间范围:</strong> {time_range}</p>
                <p><strong>统计来源:</strong> 8个摄像头统一数据 | <strong>总检测记录:</strong> {total_detections} 条</p>
            </div>
        </header>
        
        <div class="content">
            <!-- 统计概览 -->
            <section>
                <h2>📊 统计概览</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="label">总检测数</div>
                        <div class="value">{total_detections}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">检测类别</div>
                        <div class="value">{class_count}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">平均置信度</div>
                        <div class="value">{avg_confidence:.3f}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">主要类别</div>
                        <div class="value">{main_class}</div>
                    </div>
                </div>
            </section>
            
            <!-- 类别统计 -->
            <section>
                <h2>📈 全局类别统计</h2>
                <table>
                    <thead>
                        <tr>
                            <th>检测类别</th>
                            <th>检测数量</th>
                            <th>占比</th>
                            <th>可视化</th>
                            <th>平均置信度</th>
                        </tr>
                    </thead>
                    <tbody>
                        {class_stats_rows}
                    </tbody>
                </table>
            </section>
            
            <!-- 镜头分布统计 -->
            <section>
                <h2>🎥 各镜头检测分布</h2>
                <div class="feed-stats">
                    {feed_stats_cards}
                </div>
            </section>
            
            <!-- ✅ 详细记录 - 新增筛选和分页 -->
            <section>
                <h2>📋 详细检测记录</h2>
                
                <!-- ✅ 筛选器 -->
                <div class="filter-section">
                    <div class="filter-group">
                        <label for="feedFilter">筛选镜头:</label>
                        <select id="feedFilter" onchange="applyFilters()">
                            <option value="">所有镜头</option>
                            {feed_filter_options}
                        </select>
                    </div>
                    <div class="filter-group">
                        <label for="classFilter">筛选类别:</label>
                        <select id="classFilter" onchange="applyFilters()">
                            <option value="">所有类别</option>
                            {class_filter_options}
                        </select>
                    </div>
                    <div class="filter-group">
                        <label for="pageSize">每页条数:</label>
                        <select id="pageSize" onchange="applyFilters()">
                            <option value="20">20条</option>
                            <option value="50" selected>50条</option>
                            <option value="100">100条</option>
                            <option value="200">200条</option>
                        </select>
                    </div>
                </div>
                
                <!-- ✅ 统计信息 -->
                <div class="page-info" id="pageInfo"></div>
                
                <!-- ✅ 数据表格 -->
                <table id="detailTable">
                    <thead>
                        <tr>
                            <th>序号</th>
                            <th>时间戳</th>
                            <th>镜头</th>
                            <th>帧ID</th>
                            <th>检测类别</th>
                            <th>置信度</th>
                            <th>位置</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                        {detail_records}
                    </tbody>
                </table>
                
                <!-- ✅ 分页控件 -->
                <div class="pagination" id="pagination"></div>
            </section>
        </div>
        
        <footer>
            <p>此报告由皮带传送带锚杆检测系统自动生成</p>
            <p>© 2026 检测系统 | {generate_time}</p>
        </footer>
    </div>
    
    <!-- ✅ 新增JavaScript实现分页和筛选 -->
    <script>
        // 原始数据
        const allRecords = {all_records_json};
        let filteredRecords = [...allRecords];
        let currentPage = 1;
        
        // 定义强调的类别
        const alertClasses = new Set(['Other_garbage', 'bolt']);
        
        function applyFilters() {{
            const feedFilter = document.getElementById('feedFilter').value;
            const classFilter = document.getElementById('classFilter').value;
            const pageSize = parseInt(document.getElementById('pageSize').value);
            
            // 过滤数据
            filteredRecords = allRecords.filter(record => {{
                if (feedFilter && record.feed_name !== feedFilter) return false;
                if (classFilter && record.class_name !== classFilter) return false;
                return true;
            }});
            
            currentPage = 1;
            renderTable(pageSize);
            renderPagination(pageSize);
        }}
        
        function renderTable(pageSize) {{
            const tbody = document.getElementById('tableBody');
            const startIdx = (currentPage - 1) * pageSize;
            const endIdx = startIdx + pageSize;
            const pageData = filteredRecords.slice(startIdx, endIdx);
            
            tbody.innerHTML = pageData.map((record, idx) => {{
                const rowClass = alertClasses.has(record.class_name) ? 'alert-row' : '';
                return `
                    <tr class="{{rowClass}}">
                        <td>${{record.index}}</td>
                        <td>${{record.timestamp}}</td>
                        <td><span class="tag">${{record.feed_name}}</span></td>
                        <td>${{record.frame_id}}</td>
                        <td><strong>${{record.class_name}}</strong></td>
                        <td>${{record.confidence}}</td>
                        <td style="font-size:12px;">(${{record.bbox}})</td>
                    </tr>
                `;
            }}).join('');
            
            // 更新统计信息
            const total = filteredRecords.length;
            const shown = Math.min(endIdx, total);
            document.getElementById('pageInfo').textContent = 
                `显示 ${{startIdx + 1}}-${{shown}} / 总计 ${{total}} 条 | 第 ${{currentPage}}/${{Math.ceil(total / pageSize)}} 页`;
        }}
        
        function renderPagination(pageSize) {{
            const totalPages = Math.ceil(filteredRecords.length / pageSize);
            const pagination = document.getElementById('pagination');
            
            if (totalPages <= 1) {{
                pagination.innerHTML = '';
                return;
            }}
            
            let html = `<button onclick="goPage(1)" ${{currentPage === 1 ? 'disabled' : ''}}>首页</button>`;
            html += `<button onclick="goPage(${{currentPage - 1}})" ${{currentPage === 1 ? 'disabled' : ''}}>上一页</button>`;
            
            // 页码按钮
            let startPage = Math.max(1, currentPage - 2);
            let endPage = Math.min(totalPages, currentPage + 2);
            
            for (let i = startPage; i <= endPage; i++) {{
                html += `<button onclick="goPage(${{i}})" class="${{i === currentPage ? 'active' : ''}}">${{i}}</button>`;
            }}
            
            html += `<button onclick="goPage(${{currentPage + 1}})" ${{currentPage === totalPages ? 'disabled' : ''}}>下一页</button>`;
            html += `<button onclick="goPage(${{totalPages}})" ${{currentPage === totalPages ? 'disabled' : ''}}>末页</button>`;
            
            pagination.innerHTML = html;
        }}
        
        function goPage(page) {{
            currentPage = page;
            const pageSize = parseInt(document.getElementById('pageSize').value);
            renderTable(pageSize);
            renderPagination(pageSize);
            window.scrollTo(0, document.getElementById('detailTable').offsetTop - 100);
        }}
        
        // 初始化
        window.onload = function() {{
            const pageSize = parseInt(document.getElementById('pageSize').value);
            renderTable(pageSize);
            renderPagination(pageSize);
        }};
    </script>
</body>
</html>
"""
    
    def __init__(self, post_processor, logger=None):
        self.post_processor = post_processor
        self.logger = logger
    
    def generate_html_report(self, output_file=None):
        """生成HTML报告"""
        try:
            stats = self.post_processor.get_statistics()
            detections = self.post_processor.get_detection_history_copy()
            
            if not detections:
                if self.logger:
                    self.logger.warning("没有检测数据，无法生成报告")
                return None
            
            total_detections = len(detections)
            class_count = len(stats)
            main_class = max(stats.items(), key=lambda x: x[1])[0] if stats else "未知"
            avg_confidence = sum(d.confidence for d in detections) / len(detections) if detections else 0
            generate_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # ✅ 时间范围
            if detections:
                start_time = min(d.timestamp for d in detections)
                end_time = max(d.timestamp for d in detections)
                time_range = f"{start_time} 至 {end_time}"
            else:
                time_range = "无"
            
            # ✅ 生成各部分内容
            class_stats_rows = self._generate_class_stats_rows(stats, detections)
            feed_stats_cards = self._generate_feed_stats_cards(detections)
            detail_records = self._generate_detail_records(detections)
            feed_filter_options = self._generate_feed_filter_options()
            class_filter_options = self._generate_class_filter_options(stats)
            all_records_json = self._generate_records_json(detections)
            
            html_content = self.HTML_TEMPLATE.format(
                generate_time=generate_time,
                time_range=time_range,
                total_detections=total_detections,
                class_count=class_count,
                avg_confidence=avg_confidence,
                main_class=main_class,
                class_stats_rows=class_stats_rows,
                feed_stats_cards=feed_stats_cards,
                detail_records=detail_records,
                feed_filter_options=feed_filter_options,
                class_filter_options=class_filter_options,
                all_records_json=all_records_json
            )
            
            if output_file is None:
                output_dir = self.post_processor.output_dir
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(
                    output_dir,
                    f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            if self.logger:
                self.logger.info(f"HTML报告已生成: {output_file}")
            
            return output_file
        
        except Exception as e:
            if self.logger:
                self.logger.exception(f"生成HTML报告失败: {e}")
            raise
    
    def _generate_class_stats_rows(self, stats, detections):
        """生成类别统计表格行"""
        rows = []
        
        # 按检测数量从大到小排序
        for class_name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            total = len(detections)
            percentage = (count / total * 100) if total > 0 else 0
            
            # 计算该类别的平均置信度
            class_dets = [d for d in detections if d.class_name == class_name]
            avg_conf = sum(d.confidence for d in class_dets) / len(class_dets) if class_dets else 0
            
            row = f"""
                <tr>
                    <td><strong>{class_name}</strong></td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress">
                                <div class="progress-fill" style="width: {percentage}%"></div>
                            </div>
                        </div>
                    </td>
                    <td>{avg_conf:.3f}</td>
                </tr>
            """
            rows.append(row)
        
        return "".join(rows) if rows else "<tr><td colspan='5' style='text-align:center;'>暂无数据</td></tr>"
    
    def _generate_feed_stats_cards(self, detections):
        """生成各镜头统计卡片"""
        # ✅ 按 feed_id 统计
        feed_stats = defaultdict(lambda: {'count': 0, 'classes': defaultdict(int), 'confidences': []})
        
        for det in detections:
            feed_id = det.feed_id if det.feed_id is not None else 0
            feed_stats[feed_id]['count'] += 1
            feed_stats[feed_id]['classes'][det.class_name] += 1
            feed_stats[feed_id]['confidences'].append(det.confidence)
        
        cards = []
        for feed_id in range(8):
            stats = feed_stats.get(feed_id, {'count': 0, 'classes': {}, 'confidences': []})
            count = stats['count']
            
            if count == 0:
                card = f"""
                    <div class="feed-card">
                        <h3>🎥 画面{feed_id + 1}</h3>
                        <div class="feed-stats-item">
                            <span>检测数:</span>
                            <span class="stat-value">0</span>
                        </div>
                        <div class="feed-stats-item" style="color: #999;">
                            <em>暂无数据</em>
                        </div>
                    </div>
                """
            else:
                avg_conf = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0
                main_class = max(stats['classes'].items(), key=lambda x: x[1])[0] if stats['classes'] else '未知'
                
                classes_str = ", ".join([f"{name}({cnt})" for name, cnt in stats['classes'].items()])
                
                card = f"""
                    <div class="feed-card">
                        <h3>🎥 画面{feed_id + 1}</h3>
                        <div class="feed-stats-item">
                            <span>检测数:</span>
                            <span class="stat-value">{count}</span>
                        </div>
                        <div class="feed-stats-item">
                            <span>主要类别:</span>
                            <span class="stat-value">{main_class}</span>
                        </div>
                        <div class="feed-stats-item">
                            <span>平均置信度:</span>
                            <span class="stat-value">{avg_conf:.3f}</span>
                        </div>
                        <div class="feed-stats-item" style="margin-top: 10px; font-size: 12px; color: #666;">
                            <span>分布: {classes_str}</span>
                        </div>
                    </div>
                """
            
            cards.append(card)
        
        return "".join(cards)
    
    def _generate_detail_records(self, detections, limit=None):
        """生成详细记录 - 只显示前50条，其余由JS动态加载"""
        rows = []
        display_limit = limit or 50
        
        for idx, det in enumerate(detections[:display_limit], 1):
            feed_name = f"画面{det.feed_id + 1}" if det.feed_id is not None else "未知"
            alert_class = 'alert-row' if det.class_name in {'Other_garbage', 'bolt'} else ''
            row = f"""
                <tr class="{alert_class}">
                    <td>{idx}</td>
                    <td>{det.timestamp}</td>
                    <td><span class="tag">{feed_name}</span></td>
                    <td>{det.frame_id if det.frame_id is not None else '-'}</td>
                    <td><strong>{det.class_name}</strong></td>
                    <td>{det.confidence:.3f}</td>
                    <td style="font-size:12px;">({det.bbox[0]},{det.bbox[1]},{det.bbox[2]},{det.bbox[3]})</td>
                </tr>
            """
            rows.append(row)
        
        return "".join(rows) if rows else "<tr><td colspan='7' style='text-align:center;'>暂无数据</td></tr>"
    
    def _generate_feed_filter_options(self):
        """生成镜头筛选选项"""
        options = []
        for i in range(8):
            options.append(f'<option value="画面{i + 1}">画面{i + 1}</option>')
        return "\n".join(options)
    
    def _generate_class_filter_options(self, stats):
        """生成类别筛选选项"""
        options = []
        for class_name in sorted(stats.keys()):
            options.append(f'<option value="{class_name}">{class_name}</option>')
        return "\n".join(options)
    
    def _generate_records_json(self, detections):
        """生成所有记录的JSON格式，供JavaScript使用"""
        import json
        records = []
        
        for idx, det in enumerate(detections, 1):
            feed_name = f"画面{det.feed_id + 1}" if det.feed_id is not None else "未知"
            record = {
                'index': idx,
                'timestamp': det.timestamp,
                'feed_name': feed_name,
                'frame_id': str(det.frame_id) if det.frame_id is not None else '-',
                'class_name': det.class_name,
                'confidence': f"{det.confidence:.3f}",
                'bbox': f"{det.bbox[0]},{det.bbox[1]},{det.bbox[2]},{det.bbox[3]}"
            }
            records.append(record)
        
        # 返回 JSON 字符串，避免 XSS 风险
        return json.dumps(records, ensure_ascii=False)