import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout, QSlider, QSpinBox, QDialog, QScrollArea,
    QComboBox, QCheckBox, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


def remove_background(image, center, radius=30):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    bg_pixels = image[mask == 255]
    if len(bg_pixels) == 0:
        return image, None
    
    bg_color = np.mean(bg_pixels, axis=0)
    
    diff = np.abs(image.astype(np.float32) - bg_color)
    diff_gray = np.max(diff, axis=2)
    
    _, bg_mask = cv2.threshold(diff_gray.astype(np.uint8), 30, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5, 5), np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
    
    result = image.copy()
    result[bg_mask == 255] = [255, 255, 255]
    
    return result, bg_mask


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_image = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
    
    def set_cv_image(self, cv_image):
        self.cv_image = cv_image
        if cv_image is not None:
            h, w = cv_image.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(cv_image.data.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
            self.setPixmap(QPixmap.fromImage(q_image))
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.cv_image is not None:
            label_w = self.width()
            label_h = self.height()
            img_h, img_w = self.cv_image.shape[:2]
            
            scale = min(label_w / img_w, label_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            offset_x = (label_w - new_w) // 2
            offset_y = (label_h - new_h) // 2
            
            x = int((event.x() - offset_x) / scale)
            y = int((event.y() - offset_y) / scale)
            
            if 0 <= x < img_w and 0 <= y < img_h:
                self.clicked.emit(x, y)


class ZoomableLabel(QScrollArea):
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.original_pixmap = None
        
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2a2a2a;")
        self.setWidget(self.image_label)
        
        self.setMinimumSize(300, 300)
    
    def set_image(self, cv_image):
        if cv_image is None:
            return
        h, w = cv_image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(cv_image.data.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.zoom_factor = 1.0
        self.update_display()
    
    def update_display(self):
        if self.original_pixmap is None:
            return
        new_w = int(self.original_pixmap.width() * self.zoom_factor)
        new_h = int(self.original_pixmap.height() * self.zoom_factor)
        scaled = self.original_pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.resize(new_w, new_h)
    
    def zoom_in(self):
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor *= 1.25
            self.update_display()
    
    def zoom_out(self):
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor /= 1.25
            self.update_display()
    
    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.update_display()
    
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()


class ZoomCompareDialog(QDialog):
    def __init__(self, template, matched_region, aligned_region=None, defect_mask=None, missing_mask=None, extra_mask=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("放大对比 - 滚轮缩放图片 (蓝色=漏印, 红色=多印)")
        self.setMinimumSize(1400, 700)
        
        main_layout = QVBoxLayout(self)
        
        toolbar = QHBoxLayout()
        zoom_in_btn = QPushButton("放大 (+)")
        zoom_in_btn.clicked.connect(self.zoom_all_in)
        zoom_out_btn = QPushButton("缩小 (-)")
        zoom_out_btn.clicked.connect(self.zoom_all_out)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_all_zoom)
        toolbar.addWidget(QLabel("缩放控制:"))
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(reset_btn)
        toolbar.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(close_btn)
        main_layout.addLayout(toolbar)
        
        images_layout = QHBoxLayout()
        
        self.zoom_labels = []
        
        template_group = QGroupBox("模板(电子版)")
        template_layout = QVBoxLayout(template_group)
        self.template_view = ZoomableLabel("模板")
        template_layout.addWidget(self.template_view)
        images_layout.addWidget(template_group)
        self.zoom_labels.append(self.template_view)
        
        if aligned_region is not None:
            aligned_group = QGroupBox("对齐后区域")
            aligned_layout = QVBoxLayout(aligned_group)
            self.aligned_view = ZoomableLabel("对齐后区域")
            aligned_layout.addWidget(self.aligned_view)
            images_layout.addWidget(aligned_group)
            self.zoom_labels.append(self.aligned_view)
            self.aligned_view.set_image(aligned_region)
        
        matched_group = QGroupBox("匹配区域")
        matched_layout = QVBoxLayout(matched_group)
        self.matched_view = ZoomableLabel("匹配区域")
        matched_layout.addWidget(self.matched_view)
        images_layout.addWidget(matched_group)
        self.zoom_labels.append(self.matched_view)
        
        if defect_mask is not None:
            defect_group = QGroupBox("缺陷标记")
            defect_layout = QVBoxLayout(defect_group)
            self.defect_view = ZoomableLabel("缺陷标记")
            defect_layout.addWidget(self.defect_view)
            images_layout.addWidget(defect_group)
            self.zoom_labels.append(self.defect_view)
            
            compare_image = aligned_region if aligned_region is not None else matched_region
            defect_annotated = compare_image.copy()
            
            if missing_mask is not None:
                missing_contours, _ = cv2.findContours(missing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in missing_contours:
                    area = cv2.contourArea(contour)
                    if area >= 30:
                        x, y, bw, bh = cv2.boundingRect(contour)
                        cv2.rectangle(defect_annotated, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                        cv2.putText(defect_annotated, "漏印", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            if extra_mask is not None:
                extra_contours, _ = cv2.findContours(extra_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in extra_contours:
                    area = cv2.contourArea(contour)
                    if area >= 30:
                        x, y, bw, bh = cv2.boundingRect(contour)
                        cv2.rectangle(defect_annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                        cv2.putText(defect_annotated, "多印", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            self.defect_view.set_image(defect_annotated)
        
        self.template_view.set_image(template)
        self.matched_view.set_image(matched_region)
        
        main_layout.addLayout(images_layout)
    
    def zoom_all_in(self):
        for label in self.zoom_labels:
            label.zoom_in()
    
    def zoom_all_out(self):
        for label in self.zoom_labels:
            label.zoom_out()
    
    def reset_all_zoom(self):
        for label in self.zoom_labels:
            label.reset_zoom()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Equal or event.key() == Qt.Key_Plus:
            self.zoom_all_in()
        elif event.key() == Qt.Key_Minus:
            self.zoom_all_out()
        elif event.key() == Qt.Key_R:
            self.reset_all_zoom()
        elif event.key() == Qt.Key_Escape:
            self.close()


class DefectCompareDialog(QDialog):
    def __init__(self, template, scan_image, defect_regions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("缺陷抠图对比")
        self.setMinimumSize(1000, 700)
        
        self.template = template
        self.scan_image = scan_image
        self.defect_regions = defect_regions
        self.current_index = 0
        self.show_all = True
        self.auto_play = False
        self.auto_timer = None
        
        if not defect_regions:
            QMessageBox.information(self, "提示", "没有检测到缺陷")
            return
        
        main_layout = QVBoxLayout(self)
        
        toolbar = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀ 上一个")
        self.prev_btn.clicked.connect(self.prev_defect)
        toolbar.addWidget(self.prev_btn)
        
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFont(QFont("Arial", 14, QFont.Bold))
        toolbar.addWidget(self.info_label)
        
        self.next_btn = QPushButton("下一个 ▶")
        self.next_btn.clicked.connect(self.next_defect)
        toolbar.addWidget(self.next_btn)
        
        toolbar.addStretch()
        
        self.all_btn = QPushButton("显示全部")
        self.all_btn.clicked.connect(self.show_all_defects)
        self.all_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        toolbar.addWidget(self.all_btn)
        
        self.auto_btn = QPushButton("自动播放")
        self.auto_btn.clicked.connect(self.toggle_auto_play)
        self.auto_btn.setStyleSheet("background-color: #FF9800; color: white;")
        toolbar.addWidget(self.auto_btn)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(close_btn)
        
        main_layout.addLayout(toolbar)
        
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(900, 600)
        self.display_label.setStyleSheet("background-color: #2a2a2a;")
        main_layout.addWidget(self.display_label)
        
        self.speed_layout = QHBoxLayout()
        self.speed_layout.addWidget(QLabel("播放速度(秒):"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 10)
        self.speed_spin.setValue(2)
        self.speed_layout.addWidget(self.speed_spin)
        self.speed_layout.addStretch()
        main_layout.addLayout(self.speed_layout)
        
        self.update_display()
    
    def show_all_defects(self):
        self.show_all = True
        self.stop_auto_play()
        self.update_display()
    
    def prev_defect(self):
        if not self.defect_regions:
            return
        self.show_all = False
        self.current_index = (self.current_index - 1) % len(self.defect_regions)
        self.update_display()
    
    def next_defect(self):
        if not self.defect_regions:
            return
        self.show_all = False
        self.current_index = (self.current_index + 1) % len(self.defect_regions)
        self.update_display()
    
    def toggle_auto_play(self):
        if self.auto_play:
            self.stop_auto_play()
        else:
            self.start_auto_play()
    
    def start_auto_play(self):
        self.auto_play = True
        self.show_all = False
        self.auto_btn.setText("停止播放")
        self.auto_btn.setStyleSheet("background-color: #f44336; color: white;")
        from PyQt5.QtCore import QTimer
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self.auto_next)
        self.auto_timer.start(self.speed_spin.value() * 1000)
    
    def stop_auto_play(self):
        self.auto_play = False
        self.auto_btn.setText("自动播放")
        self.auto_btn.setStyleSheet("background-color: #FF9800; color: white;")
        if self.auto_timer:
            self.auto_timer.stop()
            self.auto_timer = None
    
    def auto_next(self):
        self.current_index = (self.current_index + 1) % len(self.defect_regions)
        self.update_display()
    
    def update_display(self):
        if not self.defect_regions:
            return
        
        result = self.scan_image.copy()
        
        padding = 15
        
        if self.show_all:
            for i, defect in enumerate(self.defect_regions):
                x, y, bw, bh = defect['rect']
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(self.template.shape[1], x + bw + padding)
                y2 = min(self.template.shape[0], y + bh + padding)
                
                template_patch = self.template[y1:y2, x1:x2]
                result[y1:y2, x1:x2] = template_patch
                
                if defect['type'] == '漏印':
                    box_color = (255, 0, 0)
                else:
                    box_color = (0, 0, 255)
                
                cv2.rectangle(result, (x1, y1), (x2, y2), box_color, 2)
                
                label = f"{i+1}:{defect['type']}"
                cv2.putText(result, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            self.info_label.setText(f"全部缺陷 ({len(self.defect_regions)}处)")
        else:
            defect = self.defect_regions[self.current_index]
            x, y, bw, bh = defect['rect']
            
            for i, d in enumerate(self.defect_regions):
                dx, dy, dbw, dbh = d['rect']
                if d['type'] == '漏印':
                    c = (100, 100, 255)
                else:
                    c = (255, 100, 100)
                cv2.rectangle(result, (dx, dy), (dx + dbw, dy + dbh), c, 1)
            
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(self.template.shape[1], x + bw + padding)
            y2 = min(self.template.shape[0], y + bh + padding)
            
            template_patch = self.template[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = template_patch
            
            if defect['type'] == '漏印':
                box_color = (255, 0, 0)
            else:
                box_color = (0, 0, 255)
            
            cv2.rectangle(result, (x1, y1), (x2, y2), box_color, 3)
            
            label = f"{self.current_index+1}:{defect['type']}"
            cv2.putText(result, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            self.info_label.setText(
                f"缺陷 {self.current_index+1}/{len(self.defect_regions)}  "
                f"类型:{defect['type']}  面积:{defect['area']:.0f}  "
                f"位置:({x},{y})  大小:{bw}x{bh}"
            )
        
        h, w = result.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(result.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev_defect()
        elif event.key() == Qt.Key_Right:
            self.next_defect()
        elif event.key() == Qt.Key_A:
            self.show_all_defects()
        elif event.key() == Qt.Key_Space:
            self.toggle_auto_play()
        elif event.key() == Qt.Key_Escape:
            self.close()
    
    def closeEvent(self, event):
        self.stop_auto_play()
        super().closeEvent(event)


class VisualizationDialog(QDialog):
    def __init__(self, vis_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("检测过程可视化")
        self.setMinimumSize(1200, 800)
        
        self.vis_data = vis_data
        
        main_layout = QVBoxLayout(self)
        
        toolbar = QHBoxLayout()
        zoom_in_btn = QPushButton("放大 (+)")
        zoom_in_btn.clicked.connect(self.zoom_all_in)
        zoom_out_btn = QPushButton("缩小 (-)")
        zoom_out_btn.clicked.connect(self.zoom_all_out)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_all_zoom)
        toolbar.addWidget(QLabel("缩放控制:"))
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(reset_btn)
        toolbar.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(close_btn)
        main_layout.addLayout(toolbar)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.zoom_labels = []
        
        row1_layout = QHBoxLayout()
        self.add_image_view(row1_layout, "模板原图", vis_data.get('template'))
        self.add_image_view(row1_layout, "待检测图", vis_data.get('current_image'))
        self.add_image_view(row1_layout, "对齐后图像", vis_data.get('aligned'))
        scroll_layout.addLayout(row1_layout)
        
        row2_layout = QHBoxLayout()
        self.add_image_view(row2_layout, "模板灰度", vis_data.get('template_gray'), True)
        self.add_image_view(row2_layout, "检测图灰度", vis_data.get('matched_gray'), True)
        self.add_image_view(row2_layout, "灰度差异", vis_data.get('gray_diff'), True)
        scroll_layout.addLayout(row2_layout)
        
        row3_layout = QHBoxLayout()
        self.add_image_view(row3_layout, "模板二值化", vis_data.get('template_binary'), True)
        self.add_image_view(row3_layout, "检测图二值化", vis_data.get('matched_binary'), True)
        self.add_image_view(row3_layout, "显著差异", vis_data.get('significant_diff'), True)
        scroll_layout.addLayout(row3_layout)
        
        row4_layout = QHBoxLayout()
        self.add_image_view(row4_layout, "模板前景", vis_data.get('template_fg'), True)
        self.add_image_view(row4_layout, "检测图前景", vis_data.get('matched_fg'), True)
        self.add_image_view(row4_layout, "缺陷标记结果", vis_data.get('defect_result'))
        scroll_layout.addLayout(row4_layout)
        
        row5_layout = QHBoxLayout()
        self.add_image_view(row5_layout, "漏印区域", vis_data.get('missing'), True)
        self.add_image_view(row5_layout, "多印区域", vis_data.get('extra'), True)
        scroll_layout.addLayout(row5_layout)
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
    
    def add_image_view(self, layout, title, image, is_gray=False):
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        view = ZoomableLabel(title)
        view.setMinimumSize(300, 200)
        group_layout.addWidget(view)
        layout.addWidget(group)
        self.zoom_labels.append(view)
        
        if image is not None:
            if is_gray and len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            view.set_image(image)
    
    def zoom_all_in(self):
        for label in self.zoom_labels:
            label.zoom_in()
    
    def zoom_all_out(self):
        for label in self.zoom_labels:
            label.zoom_out()
    
    def reset_all_zoom(self):
        for label in self.zoom_labels:
            label.reset_zoom()


class FeatureMatcher:
    def __init__(self):
        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            self.sift = cv2.xfeatures2d.SIFT_create()
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def align_images(self, template, image, min_matches=10):
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(template_gray, None)
        kp2, des2 = self.sift.detectAndCompute(image_gray, None)
        
        if des1 is None or des2 is None:
            return None, 0, "无法检测特征点"
        
        if len(des1) < min_matches or len(des2) < min_matches:
            return None, 0, f"特征点不足: {len(des1)}/{len(des2)}"
        
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < min_matches:
            return None, len(good_matches), f"匹配点不足: {len(good_matches)}"
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None, len(good_matches), "无法计算变换矩阵"
        
        h, w = template.shape[:2]
        aligned = cv2.warpPerspective(image, H, (w, h))
        
        return aligned, len(good_matches), "对齐成功"
    
    def get_match_visualization(self, template, image):
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(template_gray, None)
        kp2, des2 = self.sift.detectAndCompute(image_gray, None)
        
        if des1 is None or des2 is None:
            return None
        
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return None
        
        vis = cv2.drawMatches(template, kp1, image, kp2, good_matches[:20], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return vis


class TemplateMatchChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.template = None
        self.template_gray = None
        self.template_original = None
        self.image_paths = []
        self.current_image = None
        self.current_image_original = None
        self.match_threshold = 0.8
        self.matched_region = None
        self.aligned_region = None
        self.defect_mask = None
        self.missing_mask = None
        self.extra_mask = None
        self.feature_matcher = FeatureMatcher()
        self.use_feature_align = True
        self.vis_data = {}
        self.overlay_mode = 0
        self.result_image = None
        self.defect_regions = []
        self.template_bg_removed = False
        self.image_bg_removed = False
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("印刷品质量检测 - 瓦楞纸壳对版")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        template_group = QGroupBox("模板设置")
        template_layout = QVBoxLayout(template_group)
        
        self.template_label = QLabel("未加载模板")
        self.template_label.setAlignment(Qt.AlignCenter)
        self.template_label.setMinimumHeight(150)
        self.template_label.setStyleSheet("border: 2px dashed #999; background-color: #f0f0f0;")
        template_layout.addWidget(self.template_label)
        
        self.load_template_btn = QPushButton("加载模板 (temp1.jpg)")
        self.load_template_btn.clicked.connect(self.load_template)
        template_layout.addWidget(self.load_template_btn)
        
        left_layout.addWidget(template_group)
        
        images_group = QGroupBox("待检测图片")
        images_layout = QVBoxLayout(images_group)
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.select_image)
        images_layout.addWidget(self.image_list)
        
        images_btn_layout = QHBoxLayout()
        self.load_images_btn = QPushButton("加载图片")
        self.load_images_btn.clicked.connect(self.load_images)
        images_btn_layout.addWidget(self.load_images_btn)
        
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_images)
        images_btn_layout.addWidget(self.clear_btn)
        
        images_layout.addLayout(images_btn_layout)
        
        left_layout.addWidget(images_group)
        
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout(params_group)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("匹配阈值:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(80)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.80")
        threshold_layout.addWidget(self.threshold_label)
        params_layout.addLayout(threshold_layout)
        
        self.feature_align_check = QCheckBox("SIFT特征匹配对齐")
        self.feature_align_check.setChecked(True)
        self.feature_align_check.stateChanged.connect(self.on_feature_align_changed)
        params_layout.addWidget(self.feature_align_check)
        
        params_layout.addWidget(QLabel("─" * 20))
        
        binary_layout = QHBoxLayout()
        binary_layout.addWidget(QLabel("二值化块:"))
        self.binary_block_spin = QSpinBox()
        self.binary_block_spin.setRange(5, 51)
        self.binary_block_spin.setValue(21)
        self.binary_block_spin.setSingleStep(2)
        binary_layout.addWidget(self.binary_block_spin)
        params_layout.addLayout(binary_layout)
        
        binary_c_layout = QHBoxLayout()
        binary_c_layout.addWidget(QLabel("二值化C:"))
        self.binary_c_spin = QSpinBox()
        self.binary_c_spin.setRange(1, 30)
        self.binary_c_spin.setValue(10)
        binary_c_layout.addWidget(self.binary_c_spin)
        params_layout.addLayout(binary_c_layout)
        
        diff_tol_layout = QHBoxLayout()
        diff_tol_layout.addWidget(QLabel("差异容差:"))
        self.diff_tolerance_spin = QSpinBox()
        self.diff_tolerance_spin.setRange(0, 80)
        self.diff_tolerance_spin.setValue(20)
        diff_tol_layout.addWidget(self.diff_tolerance_spin)
        params_layout.addLayout(diff_tol_layout)
        
        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("最小面积:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(10, 500)
        self.min_area_spin.setValue(50)
        area_layout.addWidget(self.min_area_spin)
        params_layout.addLayout(area_layout)
        
        params_layout.addWidget(QLabel("─" * 20))
        
        params_layout.addWidget(QLabel("形态学去噪:"))
        
        morph_kernel_layout = QHBoxLayout()
        morph_kernel_layout.addWidget(QLabel("核大小:"))
        self.morph_kernel_spin = QSpinBox()
        self.morph_kernel_spin.setRange(1, 7)
        self.morph_kernel_spin.setValue(3)
        self.morph_kernel_spin.setSingleStep(2)
        morph_kernel_layout.addWidget(self.morph_kernel_spin)
        params_layout.addLayout(morph_kernel_layout)
        
        erode_layout = QHBoxLayout()
        erode_layout.addWidget(QLabel("腐蚀次数:"))
        self.erode_spin = QSpinBox()
        self.erode_spin.setRange(0, 5)
        self.erode_spin.setValue(1)
        erode_layout.addWidget(self.erode_spin)
        params_layout.addLayout(erode_layout)
        
        dilate_layout = QHBoxLayout()
        dilate_layout.addWidget(QLabel("膨胀次数:"))
        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 5)
        self.dilate_spin.setValue(1)
        dilate_layout.addWidget(self.dilate_spin)
        params_layout.addLayout(dilate_layout)
        
        params_layout.addWidget(QLabel("─" * 20))
        
        params_layout.addWidget(QLabel("背景去除(点击图片):"))
        
        bg_radius_layout = QHBoxLayout()
        bg_radius_layout.addWidget(QLabel("采样半径:"))
        self.bg_radius_spin = QSpinBox()
        self.bg_radius_spin.setRange(10, 100)
        self.bg_radius_spin.setValue(30)
        bg_radius_layout.addWidget(self.bg_radius_spin)
        params_layout.addLayout(bg_radius_layout)
        
        self.remove_template_bg_btn = QPushButton("去除模板背景")
        self.remove_template_bg_btn.clicked.connect(self.remove_template_background)
        self.remove_template_bg_btn.setEnabled(False)
        params_layout.addWidget(self.remove_template_bg_btn)
        
        self.remove_image_bg_btn = QPushButton("去除检测图背景")
        self.remove_image_bg_btn.clicked.connect(self.remove_image_background)
        self.remove_image_bg_btn.setEnabled(False)
        params_layout.addWidget(self.remove_image_bg_btn)
        
        self.reset_bg_btn = QPushButton("重置背景")
        self.reset_bg_btn.clicked.connect(self.reset_background)
        self.reset_bg_btn.setEnabled(False)
        params_layout.addWidget(self.reset_bg_btn)
        
        left_layout.addWidget(params_group)
        
        self.detect_btn = QPushButton("检测当前图片")
        self.detect_btn.clicked.connect(self.detect_current)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 12px;")
        left_layout.addWidget(self.detect_btn)
        
        self.visualize_btn = QPushButton("检测过程可视化")
        self.visualize_btn.clicked.connect(self.show_visualization)
        self.visualize_btn.setEnabled(False)
        self.visualize_btn.setStyleSheet("background-color: #9C27B0; color: white; font-size: 14px; padding: 12px;")
        left_layout.addWidget(self.visualize_btn)
        
        self.zoom_btn = QPushButton("放大对比")
        self.zoom_btn.clicked.connect(self.show_zoom_compare)
        self.zoom_btn.setEnabled(False)
        self.zoom_btn.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px; padding: 12px;")
        left_layout.addWidget(self.zoom_btn)
        
        self.overlay_btn = QPushButton("叠加显示切换")
        self.overlay_btn.clicked.connect(self.toggle_overlay)
        self.overlay_btn.setEnabled(False)
        self.overlay_btn.setStyleSheet("background-color: #FF9800; color: white; font-size: 14px; padding: 12px;")
        left_layout.addWidget(self.overlay_btn)
        
        self.defect_view_btn = QPushButton("缺陷抠图对比")
        self.defect_view_btn.clicked.connect(self.show_defect_compare)
        self.defect_view_btn.setEnabled(False)
        self.defect_view_btn.setStyleSheet("background-color: #E91E63; color: white; font-size: 14px; padding: 12px;")
        left_layout.addWidget(self.defect_view_btn)
        
        left_layout.addStretch()
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        preview_group = QGroupBox("图片预览 (点击选择背景)")
        preview_layout = QHBoxLayout(preview_group)
        
        template_container = QVBoxLayout()
        template_container.addWidget(QLabel("模板"))
        self.template_preview = ClickableLabel()
        self.template_preview.setAlignment(Qt.AlignCenter)
        self.template_preview.setMinimumSize(300, 300)
        self.template_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.template_preview.clicked.connect(self.on_template_clicked)
        template_container.addWidget(self.template_preview)
        preview_layout.addLayout(template_container)
        
        image_container = QVBoxLayout()
        image_container.addWidget(QLabel("待检测图片"))
        self.image_preview = ClickableLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(300, 300)
        self.image_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.image_preview.clicked.connect(self.on_image_clicked)
        image_container.addWidget(self.image_preview)
        preview_layout.addLayout(image_container)
        
        result_container = QVBoxLayout()
        result_label_layout = QHBoxLayout()
        result_label_layout.addWidget(QLabel("匹配结果"))
        self.overlay_mode_label = QLabel("")
        self.overlay_mode_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        result_label_layout.addWidget(self.overlay_mode_label)
        result_label_layout.addStretch()
        result_container.addLayout(result_label_layout)
        self.result_preview = QLabel()
        self.result_preview.setAlignment(Qt.AlignCenter)
        self.result_preview.setMinimumSize(300, 300)
        self.result_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        result_container.addWidget(self.result_preview)
        preview_layout.addLayout(result_container)
        
        right_layout.addWidget(preview_group)
        
        info_group = QGroupBox("检测结果")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("匹配分数:"), 0, 0)
        self.score_label = QLabel("--")
        self.score_label.setFont(QFont("Arial", 16, QFont.Bold))
        info_layout.addWidget(self.score_label, 0, 1)
        
        info_layout.addWidget(QLabel("匹配状态:"), 0, 2)
        self.status_label = QLabel("--")
        self.status_label.setFont(QFont("Arial", 16, QFont.Bold))
        info_layout.addWidget(self.status_label, 0, 3)
        
        info_layout.addWidget(QLabel("匹配位置:"), 1, 0)
        self.position_label = QLabel("--")
        info_layout.addWidget(self.position_label, 1, 1)
        
        info_layout.addWidget(QLabel("模板尺寸:"), 1, 2)
        self.size_label = QLabel("--")
        info_layout.addWidget(self.size_label, 1, 3)
        
        info_layout.addWidget(QLabel("缺陷检测:"), 2, 0)
        self.defect_label = QLabel("--")
        self.defect_label.setFont(QFont("Arial", 14, QFont.Bold))
        info_layout.addWidget(self.defect_label, 2, 1)
        
        right_layout.addWidget(info_group)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        self.load_default_template()
    
    def load_default_template(self):
        template_path = os.path.join(os.path.dirname(__file__), "img", "temp1.jpg")
        if os.path.exists(template_path):
            self.set_template(template_path)
        
        img_dir = os.path.join(os.path.dirname(__file__), "img")
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                     and f not in ["template.jpg", "temp1.jpg"]]
            self.image_paths = [os.path.join(img_dir, f) for f in sorted(images)]
            self.update_image_list()
    
    def load_template(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模板图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.set_template(file_path)
    
    def set_template(self, file_path):
        self.template = cv_imread(file_path)
        if self.template is not None:
            self.template_original = self.template.copy()
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            self.template_bg_removed = False
            self.template_label.setText(f"模板: {os.path.basename(file_path)}")
            self.template_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #e8f5e9;")
            self.display_image(self.template, self.template_preview)
            self.template_preview.set_cv_image(self.template)
            h, w = self.template.shape[:2]
            self.size_label.setText(f"{w} x {h}")
            self.remove_template_bg_btn.setEnabled(True)
            self.reset_bg_btn.setEnabled(True)
            if self.image_paths:
                self.detect_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "错误", "无法加载模板图片")
    
    def load_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择待检测图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_paths:
            self.image_paths = file_paths
            self.update_image_list()
            if self.template is not None:
                self.detect_btn.setEnabled(True)
    
    def update_image_list(self):
        self.image_list.clear()
        for path in self.image_paths:
            self.image_list.addItem(os.path.basename(path))
    
    def clear_images(self):
        self.image_paths = []
        self.image_list.clear()
        self.current_image = None
        self.detect_btn.setEnabled(False)
        self.image_preview.clear()
        self.result_preview.clear()
        self.score_label.setText("--")
        self.status_label.setText("--")
        self.position_label.setText("--")
    
    def select_image(self, item):
        index = self.image_list.row(item)
        if 0 <= index < len(self.image_paths):
            self.current_image = cv_imread(self.image_paths[index])
            self.current_image_original = self.current_image.copy()
            self.image_bg_removed = False
            self.display_image(self.current_image, self.image_preview)
            self.image_preview.set_cv_image(self.current_image)
            self.detect_btn.setEnabled(True)
            self.remove_image_bg_btn.setEnabled(True)
            self.reset_bg_btn.setEnabled(True)
            self.result_preview.clear()
            self.score_label.setText("--")
            self.status_label.setText("--")
            self.position_label.setText("--")
            self.defect_label.setText("--")
            self.overlay_mode_label.setText("")
            self.overlay_btn.setEnabled(False)
    
    def on_template_clicked(self, x, y):
        if self.template is not None:
            radius = self.bg_radius_spin.value()
            result, _ = remove_background(self.template, (x, y), radius)
            self.template = result
            self.template_bg_removed = True
            self.display_image(self.template, self.template_preview)
            self.template_preview.set_cv_image(self.template)
            QMessageBox.information(self, "背景去除", f"已去除模板背景\n采样点: ({x}, {y})")
    
    def on_image_clicked(self, x, y):
        if self.current_image is not None:
            radius = self.bg_radius_spin.value()
            result, _ = remove_background(self.current_image, (x, y), radius)
            self.current_image = result
            self.image_bg_removed = True
            self.display_image(self.current_image, self.image_preview)
            self.image_preview.set_cv_image(self.current_image)
            QMessageBox.information(self, "背景去除", f"已去除检测图背景\n采样点: ({x}, {y})")
    
    def remove_template_background(self):
        QMessageBox.information(self, "提示", "请在模板预览区域点击空白背景区域")
    
    def remove_image_background(self):
        QMessageBox.information(self, "提示", "请在待检测图片预览区域点击空白背景区域")
    
    def reset_background(self):
        if self.template_original is not None:
            self.template = self.template_original.copy()
            self.template_bg_removed = False
            self.display_image(self.template, self.template_preview)
            self.template_preview.set_cv_image(self.template)
        if self.current_image_original is not None:
            self.current_image = self.current_image_original.copy()
            self.image_bg_removed = False
            self.display_image(self.current_image, self.image_preview)
            self.image_preview.set_cv_image(self.current_image)
    
    def toggle_overlay(self):
        if self.result_image is None or self.aligned_region is None:
            return
        
        self.overlay_mode = (self.overlay_mode + 1) % 3
        
        if self.overlay_mode == 0:
            template_marked = self.template.copy()
            for defect in self.defect_regions:
                x, y, bw, bh = defect['rect']
                if defect['type'] == '漏印':
                    box_color = (255, 0, 0)
                    label_text = "漏印"
                else:
                    box_color = (0, 0, 255)
                    label_text = "多印"
                cv2.rectangle(template_marked, (x, y), (x + bw, y + bh), box_color, 2)
                cv2.putText(template_marked, label_text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            self.display_image(template_marked, self.result_preview)
            self.overlay_mode_label.setText("[样本图+缺陷框]")
        elif self.overlay_mode == 1:
            scan_marked = self.aligned_region.copy()
            for defect in self.defect_regions:
                x, y, bw, bh = defect['rect']
                if defect['type'] == '漏印':
                    box_color = (255, 0, 0)
                    label_text = "漏印"
                else:
                    box_color = (0, 0, 255)
                    label_text = "多印"
                cv2.rectangle(scan_marked, (x, y), (x + bw, y + bh), box_color, 2)
                cv2.putText(scan_marked, label_text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            self.display_image(scan_marked, self.result_preview)
            self.overlay_mode_label.setText("[扫描图+缺陷框]")
        elif self.overlay_mode == 2:
            diff = cv2.absdiff(self.template, self.aligned_region)
            diff_enhanced = cv2.convertScaleAbs(diff, alpha=3.0, beta=0)
            self.display_image(diff_enhanced, self.result_preview)
            self.overlay_mode_label.setText("[差异图]")
    
    def on_threshold_changed(self, value):
        self.match_threshold = value / 100.0
        self.threshold_label.setText(f"{self.match_threshold:.2f}")
    
    def on_feature_align_changed(self, state):
        self.use_feature_align = state == Qt.Checked
    
    def extract_print_content(self, image, block_size, c_value):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c_value
        )
        
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        content_mask = cv2.bitwise_not(binary)
        content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)
        
        combined = cv2.bitwise_or(edges, content_mask)
        
        return combined, binary
    
    def detect_print_defects(self, template, matched_region):
        block_size = self.binary_block_spin.value()
        c_value = self.binary_c_spin.value()
        min_area = self.min_area_spin.value()
        morph_kernel = self.morph_kernel_spin.value()
        erode_iter = self.erode_spin.value()
        dilate_iter = self.dilate_spin.value()
        diff_tolerance = self.diff_tolerance_spin.value()
        
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        matched_gray = cv2.cvtColor(matched_region, cv2.COLOR_BGR2GRAY)
        
        template_blur = cv2.GaussianBlur(template_gray, (5, 5), 1)
        matched_blur = cv2.GaussianBlur(matched_gray, (5, 5), 1)
        
        template_binary = cv2.adaptiveThreshold(
            template_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c_value
        )
        matched_binary = cv2.adaptiveThreshold(
            matched_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c_value
        )
        
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        
        if erode_iter > 0:
            matched_binary = cv2.erode(matched_binary, kernel, iterations=erode_iter)
        
        if dilate_iter > 0:
            matched_binary = cv2.dilate(matched_binary, kernel, iterations=dilate_iter)
        
        template_fg = cv2.bitwise_not(template_binary)
        matched_fg = cv2.bitwise_not(matched_binary)
        
        template_fg = cv2.morphologyEx(template_fg, cv2.MORPH_CLOSE, kernel)
        matched_fg = cv2.morphologyEx(matched_fg, cv2.MORPH_CLOSE, kernel)
        
        gray_diff = cv2.absdiff(template_blur, matched_blur)
        
        signed_diff = template_blur.astype(np.float32) - matched_blur.astype(np.float32)
        
        missing_mask = ((signed_diff > diff_tolerance) & (template_fg > 0) & (matched_fg == 0)).astype(np.uint8) * 255
        extra_mask = ((signed_diff < -diff_tolerance) & (matched_fg > 0) & (template_fg == 0)).astype(np.uint8) * 255
        
        missing = cv2.bitwise_and(missing_mask, template_fg)
        missing = cv2.bitwise_and(missing, cv2.bitwise_not(matched_fg))
        missing = cv2.morphologyEx(missing, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        missing = cv2.morphologyEx(missing, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        extra = cv2.bitwise_and(extra_mask, matched_fg)
        extra = cv2.bitwise_and(extra, cv2.bitwise_not(template_fg))
        extra = cv2.morphologyEx(extra, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        extra = cv2.morphologyEx(extra, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        significant_diff = cv2.bitwise_or(missing, extra)
        
        defect_regions = []
        
        missing_contours, _ = cv2.findContours(missing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in missing_contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, bw, bh = cv2.boundingRect(contour)
                rect_area = bw * bh
                if rect_area > 0:
                    fill_ratio = area / rect_area
                else:
                    fill_ratio = 1.0
                if fill_ratio < 0.1:
                    continue
                defect_regions.append({
                    'rect': (x, y, bw, bh),
                    'area': area,
                    'type': '漏印'
                })
        
        extra_contours, _ = cv2.findContours(extra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in extra_contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, bw, bh = cv2.boundingRect(contour)
                rect_area = bw * bh
                if rect_area > 0:
                    fill_ratio = area / rect_area
                else:
                    fill_ratio = 1.0
                if fill_ratio < 0.1:
                    continue
                defect_regions.append({
                    'rect': (x, y, bw, bh),
                    'area': area,
                    'type': '多印'
                })
        
        defect_regions = self.merge_overlapping_boxes(defect_regions)
        
        combined_defects = cv2.bitwise_or(missing, extra)
        
        self.vis_data['template_gray'] = template_gray
        self.vis_data['matched_gray'] = matched_gray
        self.vis_data['template_binary'] = template_binary
        self.vis_data['matched_binary'] = matched_binary
        self.vis_data['template_fg'] = template_fg
        self.vis_data['matched_fg'] = matched_fg
        self.vis_data['missing'] = missing
        self.vis_data['extra'] = extra
        self.vis_data['gray_diff'] = gray_diff
        self.vis_data['significant_diff'] = significant_diff
        
        return defect_regions, combined_defects, missing, extra
    
    def merge_overlapping_boxes(self, regions):
        if len(regions) <= 1:
            return regions
        
        def compute_iou(r1, r2):
            x1, y1, w1, h1 = r1['rect']
            x2, y2, w2, h2 = r2['rect']
            
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            
            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - inter
            
            return inter / union if union > 0 else 0.0
        
        missing_regions = [r for r in regions if r['type'] == '漏印']
        extra_regions = [r for r in regions if r['type'] == '多印']
        
        result = []
        
        for group in [missing_regions, extra_regions]:
            if not group:
                continue
            
            group.sort(key=lambda r: r['area'], reverse=True)
            
            keep = []
            suppressed = [False] * len(group)
            
            for i in range(len(group)):
                if suppressed[i]:
                    continue
                keep.append(group[i])
                for j in range(i + 1, len(group)):
                    if suppressed[j]:
                        continue
                    if compute_iou(group[i], group[j]) > 0.3:
                        suppressed[j] = True
            
            result.extend(keep)
        
        return result
    
    def detect_defects(self, template, matched_region):
        return self.detect_print_defects(template, matched_region)
    
    def detect_current(self):
        if self.template is None:
            QMessageBox.warning(self, "提示", "请先加载模板")
            return
        
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先选择待检测图片")
            return
        
        self.matched_region = None
        self.aligned_region = None
        self.defect_mask = None
        self.missing_mask = None
        self.extra_mask = None
        self.zoom_btn.setEnabled(False)
        self.visualize_btn.setEnabled(False)
        self.defect_view_btn.setEnabled(False)
        
        self.vis_data = {
            'template': self.template.copy(),
            'current_image': self.current_image.copy()
        }
        
        result_image = self.current_image.copy()
        compare_region = None
        align_msg = ""
        max_loc = (0, 0)
        
        if self.use_feature_align:
            aligned, match_count, msg = self.feature_matcher.align_images(
                self.template, self.current_image
            )
            
            if aligned is not None:
                self.aligned_region = aligned.copy()
                compare_region = aligned
                result_image = aligned.copy()
                align_msg = f" (SIFT匹配: {match_count}点)"
                
                self.vis_data['aligned'] = aligned.copy()
                
                status = "特征对齐成功"
                color = (255, 165, 0)
                self.status_label.setStyleSheet("color: #FF8C00; font-weight: bold;")
                match_score = match_count / 100.0
            else:
                align_msg = f" (特征匹配失败: {msg})"
        
        if compare_region is None:
            image_gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(image_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            match_score = max_val
            top_left = max_loc
            h, w = self.template_gray.shape
            
            if match_score >= self.match_threshold:
                status = "模板匹配成功"
                color = (0, 255, 0)
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                
                matched_region = self.current_image[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
                
                if matched_region.shape[:2] == self.template.shape[:2]:
                    self.matched_region = matched_region.copy()
                    compare_region = matched_region
                    
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(result_image, top_left, bottom_right, color, 3)
            else:
                status = "匹配失败"
                color = (0, 0, 255)
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                self.defect_label.setText("--")
                self.display_image(result_image, self.result_preview)
                self.score_label.setText(f"{match_score*100:.1f}%")
                self.status_label.setText(status + align_msg)
                return
        
        if compare_region is not None and compare_region.shape[:2] == self.template.shape[:2]:
            defect_regions, defect_mask, missing, extra = self.detect_defects(self.template, compare_region)
            self.defect_mask = defect_mask.copy()
            self.missing_mask = missing.copy()
            self.extra_mask = extra.copy()
            self.defect_regions = defect_regions
            
            self.zoom_btn.setEnabled(True)
            self.visualize_btn.setEnabled(True)
            self.defect_view_btn.setEnabled(True)
            
            self.vis_data['defect_result'] = result_image.copy()
            
            missing_count = sum(1 for d in defect_regions if d['type'] == '漏印')
            extra_count = sum(1 for d in defect_regions if d['type'] == '多印')
            
            if missing_count > 0 or extra_count > 0:
                defect_text = f"漏印:{missing_count} 多印:{extra_count}"
                self.defect_label.setText(defect_text)
                self.defect_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.defect_label.setText("印刷正常")
                self.defect_label.setStyleSheet("color: green; font-weight: bold;")
            
            for defect in defect_regions:
                x, y, bw, bh = defect['rect']
                if defect['type'] == '漏印':
                    box_color = (255, 0, 0)
                    label_text = f"漏印"
                else:
                    box_color = (0, 0, 255)
                    label_text = f"多印"
                
                if self.aligned_region is not None:
                    cv2.rectangle(result_image, (x, y), (x + bw, y + bh), box_color, 2)
                    cv2.putText(result_image, label_text, 
                               (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                elif self.matched_region is not None:
                    global_x = max_loc[0] + x
                    global_y = max_loc[1] + y
                    cv2.rectangle(result_image, (global_x, global_y), (global_x + bw, global_y + bh), box_color, 2)
                    cv2.putText(result_image, label_text, 
                               (global_x, global_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        self.result_image = result_image.copy()
        self.overlay_mode = 0
        self.overlay_mode_label.setText("")
        
        if self.aligned_region is not None:
            self.overlay_btn.setEnabled(True)
        
        self.display_image(result_image, self.result_preview)
        self.score_label.setText(f"{match_score*100:.1f}%")
        self.status_label.setText(status + align_msg)
        self.position_label.setText("特征对齐" if self.aligned_region is not None else f"({max_loc[0]}, {max_loc[1]})")
    
    def show_visualization(self):
        if self.vis_data:
            dialog = VisualizationDialog(self.vis_data, self)
            dialog.exec_()
    
    def show_defect_compare(self):
        if self.aligned_region is None or not self.defect_regions:
            QMessageBox.information(self, "提示", "请先进行检测，且存在缺陷")
            return
        dialog = DefectCompareDialog(
            self.template, self.aligned_region, self.defect_regions, self
        )
        dialog.exec_()
    
    def show_zoom_compare(self):
        if self.template is not None and (self.matched_region is not None or self.aligned_region is not None):
            compare_region = self.aligned_region if self.aligned_region is not None else self.matched_region
            dialog = ZoomCompareDialog(
                self.template, self.matched_region, self.aligned_region, 
                self.defect_mask, self.missing_mask, self.extra_mask, self
            )
            dialog.exec_()
    
    def display_image(self, cv_image, label):
        if cv_image is None:
            return
        h, w = cv_image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        if hasattr(label, 'set_cv_image'):
            label.set_cv_image(cv_image)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TemplateMatchChecker()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
