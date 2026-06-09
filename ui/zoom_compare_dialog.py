from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSplitter, QSizePolicy, QWidget
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, QPoint, QRectF, pyqtSignal
from ui.image_viewer import ImageViewer
import cv2
import numpy as np


class DefectCompareWidget(QWidget):
    """缺陷区域对比组件，循环切换模板和扫描图"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.template_image = None
        self.matched_image = None
        self.current_rect = (0, 0, 0, 0)
        self.showing_template = True
        self.padding = 20
        self.setMinimumSize(300, 300)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.toggle_view)
        self._cycling = False
        self._img_data_t = None
        self._img_data_m = None
    
    def set_defect(self, template_image, matched_image, rect):
        self.template_image = template_image
        self.matched_image = matched_image
        self.current_rect = rect
        self.showing_template = True
        self._stop_cycle()
        self.start_cycle(500)
    
    def start_cycle(self, interval=500):
        self._cycling = True
        self._timer.start(interval)
    
    def stop_cycle(self):
        self._stop_cycle()
    
    def _stop_cycle(self):
        self._cycling = False
        self._timer.stop()
    
    def toggle_view(self):
        self.showing_template = not self.showing_template
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        x, y, w, h = self.current_rect
        pad = self.padding
        
        if self.showing_template and self.template_image is not None:
            img = self.template_image
            label = "模板(电子版)"
            border_color = QColor(0, 150, 255)
        elif not self.showing_template and self.matched_image is not None:
            img = self.matched_image
            label = "扫描图(印刷版)"
            border_color = QColor(255, 50, 50)
        else:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "点击缺陷区域查看对比")
            return
        
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return
        
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        crop = np.ascontiguousarray(crop)
        
        ch, cw = crop.shape[:2]
        bytes_per_line = 3 * cw
        q_image = QImage(crop.data.tobytes(), cw, ch, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image.copy())
        
        vw = self.width() - 20
        vh = self.height() - 50
        if vw <= 0 or vh <= 0:
            return
        scale = min(vw / cw, vh / ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        scaled = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        ox = (self.width() - new_w) // 2
        oy = (self.height() - new_h) // 2 + 10
        
        painter.drawPixmap(ox, oy, scaled)
        
        pen = QPen(border_color, 3)
        painter.setPen(pen)
        rx = ox + int((x - x1) * scale)
        ry = oy + int((y - y1) * scale)
        rw = int(w * scale)
        rh = int(h * scale)
        painter.drawRect(rx, ry, rw, rh)
        
        font = QFont()
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(border_color)
        painter.drawText(10, 25, label)
        
        font.setPixelSize(14)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(200, 200, 200))
        status = "▶ 循环中" if self._cycling else ("[模板]" if self.showing_template else "[扫描图]")
        painter.drawText(self.width() - 150, 25, status)


class ZoomCompareDialog(QDialog):
    """全屏放大对比对话框，支持关联缩放拖动和缺陷对比"""
    
    def __init__(self, template, matched_region, aligned_region=None,
                 defect_mask=None, missing_mask=None, extra_mask=None,
                 defect_regions=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("放大对比 - 左键拖动 | 滚轮缩放 | 右键点击缺陷对比")
        self.showFullScreen()
        
        self.template = template
        self.matched_region = matched_region
        self.aligned_region = aligned_region
        self.defect_regions = defect_regions or []
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        
        zoom_in_btn = QPushButton("放大 (+)")
        zoom_in_btn.clicked.connect(self.zoom_all_in)
        zoom_out_btn = QPushButton("缩小 (-)")
        zoom_out_btn.clicked.connect(self.zoom_all_out)
        fit_btn = QPushButton("适应窗口")
        fit_btn.clicked.connect(self.fit_all)
        reset_btn = QPushButton("1:1原始大小")
        reset_btn.clicked.connect(self.reset_all)
        
        self.link_cb = QPushButton("🔗 关联同步")
        self.link_cb.setCheckable(True)
        self.link_cb.setChecked(True)
        self.link_cb.clicked.connect(self.toggle_link)
        self.link_cb.setStyleSheet("QPushButton:checked { background-color: #4CAF50; color: white; }")
        
        self.cycle_btn = QPushButton("▶ 循环对比")
        self.cycle_btn.setCheckable(True)
        self.cycle_btn.setChecked(False)
        self.cycle_btn.clicked.connect(self.toggle_cycle)
        self.cycle_btn.setStyleSheet("QPushButton:checked { background-color: #FF9800; color: white; }")
        
        toolbar.addWidget(QLabel("缩放:"))
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(fit_btn)
        toolbar.addWidget(reset_btn)
        toolbar.addWidget(self.link_cb)
        toolbar.addWidget(self.cycle_btn)
        toolbar.addStretch()
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #aaa; font-size: 13px;")
        toolbar.addWidget(self.zoom_label)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("background-color: #d32f2f; color: white;")
        toolbar.addWidget(close_btn)
        main_layout.addLayout(toolbar)
        
        content_splitter = QSplitter(Qt.Horizontal)
        
        compare_image = aligned_region if aligned_region is not None else matched_region
        
        defect_rects = []
        for d in self.defect_regions:
            x, y, w, h = d['rect']
            defect_rects.append((x, y, w, h, d['type']))
        
        self.template_view = ImageViewer("模板(电子版)")
        self.template_view.set_image(template)
        self.template_view.set_defect_rects(defect_rects)
        self.template_view.zoom_changed.connect(self.on_zoom_changed)
        self.template_view.pan_changed.connect(self.on_pan_changed)
        self.template_view.defect_clicked.connect(self.on_defect_clicked)
        
        template_group = QGroupBox("模板(电子版)")
        template_group.setStyleSheet("QGroupBox { font-weight: bold; color: #4FC3F7; }")
        tl = QVBoxLayout(template_group)
        tl.setContentsMargins(2, 18, 2, 2)
        tl.addWidget(self.template_view)
        content_splitter.addWidget(template_group)
        
        self.matched_view = ImageViewer("扫描图(印刷版)")
        self.matched_view.set_image(compare_image)
        self.matched_view.set_defect_rects(defect_rects)
        self.matched_view.zoom_changed.connect(self.on_zoom_changed)
        self.matched_view.pan_changed.connect(self.on_pan_changed)
        self.matched_view.defect_clicked.connect(self.on_defect_clicked)
        
        matched_group = QGroupBox("扫描图(印刷版)")
        matched_group.setStyleSheet("QGroupBox { font-weight: bold; color: #EF5350; }")
        ml = QVBoxLayout(matched_group)
        ml.setContentsMargins(2, 18, 2, 2)
        ml.addWidget(self.matched_view)
        content_splitter.addWidget(matched_group)
        
        self.defect_compare = DefectCompareWidget()
        defect_group = QGroupBox("缺陷对比 (右键点击缺陷)")
        defect_group.setStyleSheet("QGroupBox { font-weight: bold; color: #FF9800; }")
        dl = QVBoxLayout(defect_group)
        dl.setContentsMargins(2, 18, 2, 2)
        dl.addWidget(self.defect_compare)
        content_splitter.addWidget(defect_group)
        
        content_splitter.setSizes([self.width() * 3 // 8, self.width() * 3 // 8, self.width() // 4])
        
        main_layout.addWidget(content_splitter, 1)
        
        self._updating = False
    
    def on_zoom_changed(self, zoom_factor, px, py):
        if self._updating:
            return
        self._updating = True
        sender = self.sender()
        img_x = (px - sender.offset.x()) / zoom_factor
        img_y = (py - sender.offset.y()) / zoom_factor
        for view in [self.template_view, self.matched_view]:
            if view != sender:
                view.zoom_factor = zoom_factor
                cx = view.width() / 2
                cy = view.height() / 2
                view.offset = QPoint(
                    int(cx - img_x * zoom_factor),
                    int(cy - img_y * zoom_factor)
                )
                view.update()
        self._updating = False
        self.zoom_label.setText(f"{zoom_factor*100:.0f}%")
    
    def on_pan_changed(self, dx, dy):
        if self._updating:
            return
        self._updating = True
        for view in [self.template_view, self.matched_view]:
            if view != self.sender():
                view.offset += QPoint(int(dx), int(dy))
                view.update()
        self._updating = False
    
    def on_defect_clicked(self, x, y, w, h):
        compare_image = self.aligned_region if self.aligned_region is not None else self.matched_region
        self.defect_compare.set_defect(self.template, compare_image, (x, y, w, h))
        if self.cycle_btn.isChecked():
            self.defect_compare.start_cycle()
    
    def toggle_link(self, checked):
        self.template_view.linked = checked
        self.matched_view.linked = checked
    
    def toggle_cycle(self, checked):
        if checked:
            self.defect_compare.start_cycle()
        else:
            self.defect_compare.stop_cycle()
    
    def zoom_all_in(self):
        center = QPoint(self.template_view.width() // 2, self.template_view.height() // 2)
        self.template_view.zoom_at(center, 1.3)
        self.matched_view.zoom_at(center, 1.3)
    
    def zoom_all_out(self):
        center = QPoint(self.template_view.width() // 2, self.template_view.height() // 2)
        self.template_view.zoom_at(center, 1.0 / 1.3)
        self.matched_view.zoom_at(center, 1.0 / 1.3)
    
    def fit_all(self):
        self.template_view.fit_to_view()
        self.matched_view.fit_to_view()
    
    def reset_all(self):
        if self.template_view.original_pixmap:
            self.template_view.zoom_factor = 1.0
            self.template_view.offset = QPoint(0, 0)
            self.template_view.update()
        if self.matched_view.original_pixmap:
            self.matched_view.zoom_factor = 1.0
            self.matched_view.offset = QPoint(0, 0)
            self.matched_view.update()
        self.zoom_label.setText("100%")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F:
            self.fit_all()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_all_in()
        elif event.key() == Qt.Key_Minus:
            self.zoom_all_out()
