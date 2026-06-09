import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout, QSplitter, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from skimage.metrics import structural_similarity as ssim


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


class QualityDetector:
    def __init__(self):
        self.template = None
        self.template_gray = None
        self.template_hsv = None
        self.template_lab = None
        self.params = {
            'diff_threshold': 25,
            'color_weight': 50,
            'edge_weight': 300,
            'defect_weight': 500,
            'min_defect_area': 10,
            'blur_size': 3,
            'ssim_threshold': 0.95,
            'edge_low': 30,
            'edge_high': 100,
            'use_ssim': True,
            'use_edge': True,
            'use_color_diff': True,
            'scratch_detect': True,
            'use_template_match': True,
            'template_match_threshold': 0.8,
            'match_method': 5
        }
        
    def load_template(self, template_path):
        self.template = cv_imread(template_path)
        if self.template is None:
            return False
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.template_hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        self.template_lab = cv2.cvtColor(self.template, cv2.COLOR_BGR2LAB)
        return True
    
    def set_param(self, name, value):
        if name in self.params:
            self.params[name] = value
    
    def template_match_defects(self, template_gray, image_gray, template_color, image_color):
        method = self.params['match_method']
        threshold = self.params['template_match_threshold']
        
        result = cv2.matchTemplate(image_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            match_score = 1 - min_val
            top_left = min_loc
        else:
            match_score = max_val
            top_left = max_loc
        
        h, w = template_gray.shape
        aligned_image = image_color[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
        aligned_gray = image_gray[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
        
        if aligned_image.shape[:2] != template_color.shape[:2]:
            aligned_image = cv2.resize(aligned_image, (w, h))
            aligned_gray = cv2.resize(aligned_gray, (w, h))
        
        diff = cv2.absdiff(template_gray, aligned_gray)
        _, defect_mask = cv2.threshold(diff, self.params['diff_threshold'], 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
        
        return defect_mask, match_score, aligned_image, aligned_gray
    
    def multi_scale_template_match(self, template_gray, image_gray):
        best_score = 0
        best_scale = 1.0
        best_location = (0, 0)
        
        for scale in np.linspace(0.8, 1.2, 9):
            resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale)
            
            if resized_template.shape[0] > image_gray.shape[0] or resized_template.shape[1] > image_gray.shape[1]:
                continue
            
            result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_scale = scale
                best_location = max_loc
        
        return best_score, best_scale, best_location
    
    def detect_scratches(self, gray1, gray2):
        edges1 = cv2.Canny(gray1, self.params['edge_low'], self.params['edge_high'])
        edges2 = cv2.Canny(gray2, self.params['edge_low'], self.params['edge_high'])
        
        edges_diff = cv2.absdiff(edges1, edges2)
        
        kernel_h = np.ones((1, 15), np.uint8)
        kernel_v = np.ones((15, 1), np.uint8)
        
        dilated_h = cv2.dilate(edges_diff, kernel_h, iterations=1)
        dilated_v = cv2.dilate(edges_diff, kernel_v, iterations=1)
        
        scratches = cv2.bitwise_or(dilated_h, dilated_v)
        scratches = cv2.morphologyEx(scratches, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        return scratches
    
    def detect_color_defects(self, lab1, lab2):
        l_diff = cv2.absdiff(lab1[:,:,0], lab2[:,:,0])
        a_diff = cv2.absdiff(lab1[:,:,1], lab2[:,:,1])
        b_diff = cv2.absdiff(lab1[:,:,2], lab2[:,:,2])
        
        l_thresh = cv2.threshold(l_diff, 15, 255, cv2.THRESH_BINARY)[1]
        a_thresh = cv2.threshold(a_diff, 10, 255, cv2.THRESH_BINARY)[1]
        b_thresh = cv2.threshold(b_diff, 10, 255, cv2.THRESH_BINARY)[1]
        
        color_defects = cv2.bitwise_or(l_thresh, cv2.bitwise_or(a_thresh, b_thresh))
        
        kernel = np.ones((3, 3), np.uint8)
        color_defects = cv2.morphologyEx(color_defects, cv2.MORPH_CLOSE, kernel)
        
        return color_defects
    
    def detect_ssim_regions(self, gray1, gray2):
        score, diff = ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")
        
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh, score
    
    def detect(self, image_path):
        if self.template is None:
            return None, "模板未加载"
        
        image = cv_imread(image_path)
        if image is None:
            return None, "无法读取图片"
        
        results = {}
        
        h, w = self.template.shape[:2]
        image_gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        match_score = 1.0
        aligned = False
        
        if self.params['use_template_match']:
            if image_gray_full.shape[0] >= h and image_gray_full.shape[1] >= w:
                try:
                    defect_mask, match_score, aligned_image, aligned_gray = self.template_match_defects(
                        self.template_gray, image_gray_full, self.template, image
                    )
                    image_resized = aligned_image
                    gray = aligned_gray
                    aligned = True
                    results['template_match_score'] = match_score
                    results['template_aligned'] = True
                except Exception as e:
                    image_resized = cv2.resize(image, (w, h))
                    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                    results['template_aligned'] = False
            else:
                image_resized = cv2.resize(image, (w, h))
                gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                results['template_aligned'] = False
        else:
            image_resized = cv2.resize(image, (w, h))
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            results['template_aligned'] = False
        
        hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
        
        blur_size = self.params['blur_size']
        if blur_size % 2 == 0:
            blur_size += 1
        template_blur = cv2.GaussianBlur(self.template_gray, (blur_size, blur_size), 0)
        image_blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        diff = cv2.absdiff(template_blur, image_blur)
        threshold = self.params['diff_threshold']
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        combined_defects = thresh.copy()
        
        if self.params['use_ssim']:
            ssim_defects, ssim_score = self.detect_ssim_regions(template_blur, image_blur)
            combined_defects = cv2.bitwise_or(combined_defects, ssim_defects)
            results['ssim_score'] = ssim_score
            results['ssim_defects'] = ssim_defects
        
        if self.params['use_color_diff']:
            color_defects = self.detect_color_defects(self.template_lab, lab)
            combined_defects = cv2.bitwise_or(combined_defects, color_defects)
            results['color_defects'] = color_defects
        
        if self.params['scratch_detect']:
            scratches = self.detect_scratches(self.template_gray, gray)
            combined_defects = cv2.bitwise_or(combined_defects, scratches)
            results['scratches'] = scratches
        
        combined_defects = cv2.morphologyEx(combined_defects, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        defect_pixels = np.sum(combined_defects > 0)
        total_pixels = w * h
        defect_ratio = defect_pixels / total_pixels
        results['defect_ratio'] = defect_ratio
        results['defect_image'] = combined_defects
        
        contours, _ = cv2.findContours(combined_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.params['min_defect_area']
        defect_regions = []
        annotated_image = image_resized.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, bw, bh = cv2.boundingRect(contour)
                defect_regions.append({
                    'rect': (x, y, bw, bh),
                    'area': area,
                    'center': (x + bw//2, y + bh//2)
                })
                cv2.rectangle(annotated_image, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(annotated_image, f'{int(area)}px', (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        results['defect_regions'] = defect_regions
        results['annotated_image'] = annotated_image
        results['defect_count'] = len(defect_regions)
        
        hist_template = cv2.calcHist([self.template_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_image = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_image, hist_image, 0, 1, cv2.NORM_MINMAX)
        color_similarity = cv2.compareHist(hist_template, hist_image, cv2.HISTCMP_CORREL)
        results['color_similarity'] = color_similarity
        
        template_edges = cv2.Canny(self.template_gray, 50, 150)
        image_edges = cv2.Canny(gray, 50, 150)
        edge_diff = cv2.absdiff(template_edges, image_edges)
        edge_defect_ratio = np.sum(edge_diff > 0) / total_pixels
        results['edge_defect_ratio'] = edge_defect_ratio
        
        mse = np.mean((self.template.astype(float) - image_resized.astype(float)) ** 2)
        results['mse'] = mse
        
        template_brightness = np.mean(self.template_gray)
        image_brightness = np.mean(gray)
        brightness_diff = abs(template_brightness - image_brightness) / 255.0
        results['brightness_diff'] = brightness_diff
        
        dw = self.params['defect_weight']
        cw = self.params['color_weight']
        ew = self.params['edge_weight']
        
        ssim_penalty = 0
        if 'ssim_score' in results:
            ssim_penalty = (1 - results['ssim_score']) * 200
        
        if defect_ratio < 0.02 and color_similarity > 0.85 and edge_defect_ratio < 0.05:
            results['quality'] = "合格"
            results['score'] = 100 - (defect_ratio * dw + (1 - color_similarity) * cw + edge_defect_ratio * ew + ssim_penalty)
        elif defect_ratio < 0.05 and color_similarity > 0.70 and edge_defect_ratio < 0.10:
            results['quality'] = "轻微缺陷"
            results['score'] = 80 - (defect_ratio * dw * 0.8 + (1 - color_similarity) * cw * 0.8 + edge_defect_ratio * ew * 0.8 + ssim_penalty * 0.8)
        else:
            results['quality'] = "不合格"
            results['score'] = max(0, 60 - (defect_ratio * dw * 0.6 + (1 - color_similarity) * cw * 0.6 + edge_defect_ratio * ew * 0.6 + ssim_penalty * 0.6))
        
        results['score'] = max(0, min(100, results['score']))
        results['diff_image'] = diff
        results['edge_diff_image'] = edge_diff
        results['resized_image'] = image_resized
        
        return results, "检测完成"


class PrintQualityChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = QualityDetector()
        self.image_paths = []
        self.current_result = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("印刷质量检测系统 - 单张检测模式")
        self.setGeometry(100, 100, 1500, 950)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        template_group = QGroupBox("模板设置")
        template_layout = QVBoxLayout(template_group)
        
        self.template_label = QLabel("未加载模板")
        self.template_label.setAlignment(Qt.AlignCenter)
        self.template_label.setMinimumHeight(150)
        self.template_label.setStyleSheet("border: 2px dashed #999; background-color: #f0f0f0;")
        template_layout.addWidget(self.template_label)
        
        self.load_template_btn = QPushButton("加载模板图片")
        self.load_template_btn.clicked.connect(self.load_template)
        template_layout.addWidget(self.load_template_btn)
        
        left_layout.addWidget(template_group)
        
        images_group = QGroupBox("待检测图片列表")
        images_layout = QVBoxLayout(images_group)
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.select_image)
        images_layout.addWidget(self.image_list)
        
        images_btn_layout = QHBoxLayout()
        self.load_images_btn = QPushButton("加载图片")
        self.load_images_btn.clicked.connect(self.load_images)
        images_btn_layout.addWidget(self.load_images_btn)
        
        self.clear_images_btn = QPushButton("清空")
        self.clear_images_btn.clicked.connect(self.clear_images)
        images_btn_layout.addWidget(self.clear_images_btn)
        
        images_layout.addLayout(images_btn_layout)
        
        left_layout.addWidget(images_group)
        
        params_group = QGroupBox("检测参数")
        params_layout = QGridLayout(params_group)
        
        row = 0
        params_layout.addWidget(QLabel("差异阈值:"), row, 0)
        self.diff_threshold_spin = QSpinBox()
        self.diff_threshold_spin.setRange(1, 100)
        self.diff_threshold_spin.setValue(25)
        self.diff_threshold_spin.valueChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.diff_threshold_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("模糊核大小:"), row, 0)
        self.blur_size_spin = QSpinBox()
        self.blur_size_spin.setRange(1, 21)
        self.blur_size_spin.setValue(3)
        self.blur_size_spin.setSingleStep(2)
        self.blur_size_spin.valueChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.blur_size_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("最小缺陷面积:"), row, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(10, 5000)
        self.min_area_spin.setValue(50)
        self.min_area_spin.valueChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.min_area_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("边缘检测低阈值:"), row, 0)
        self.edge_low_spin = QSpinBox()
        self.edge_low_spin.setRange(10, 200)
        self.edge_low_spin.setValue(30)
        self.edge_low_spin.valueChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.edge_low_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("边缘检测高阈值:"), row, 0)
        self.edge_high_spin = QSpinBox()
        self.edge_high_spin.setRange(50, 300)
        self.edge_high_spin.setValue(100)
        self.edge_high_spin.valueChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.edge_high_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("缺陷权重:"), row, 0)
        self.defect_weight_spin = QSpinBox()
        self.defect_weight_spin.setRange(100, 1000)
        self.defect_weight_spin.setValue(500)
        self.defect_weight_spin.valueChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.defect_weight_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("颜色权重:"), row, 0)
        self.color_weight_spin = QSpinBox()
        self.color_weight_spin.setRange(10, 200)
        self.color_weight_spin.setValue(50)
        self.color_weight_spin.valueChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.color_weight_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("边缘权重:"), row, 0)
        self.edge_weight_spin = QSpinBox()
        self.edge_weight_spin.setRange(100, 500)
        self.edge_weight_spin.setValue(300)
        self.edge_weight_spin.valueChanged.connect(self.on_param_changed)
        params_layout.addWidget(self.edge_weight_spin, row, 1)
        
        left_layout.addWidget(params_group)
        
        methods_group = QGroupBox("检测方法")
        methods_layout = QVBoxLayout(methods_group)
        
        self.template_match_check = QCheckBox("模板匹配对齐")
        self.template_match_check.setChecked(True)
        self.template_match_check.stateChanged.connect(self.on_param_changed)
        methods_layout.addWidget(self.template_match_check)
        
        self.ssim_check = QCheckBox("SSIM结构相似度检测")
        self.ssim_check.setChecked(True)
        self.ssim_check.stateChanged.connect(self.on_param_changed)
        methods_layout.addWidget(self.ssim_check)
        
        self.color_diff_check = QCheckBox("LAB颜色差异检测")
        self.color_diff_check.setChecked(True)
        self.color_diff_check.stateChanged.connect(self.on_param_changed)
        methods_layout.addWidget(self.color_diff_check)
        
        self.scratch_check = QCheckBox("划痕检测")
        self.scratch_check.setChecked(True)
        self.scratch_check.stateChanged.connect(self.on_param_changed)
        methods_layout.addWidget(self.scratch_check)
        
        left_layout.addWidget(methods_group)
        
        self.detect_btn = QPushButton("检测当前图片")
        self.detect_btn.clicked.connect(self.detect_current)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 12px;")
        left_layout.addWidget(self.detect_btn)
        
        self.re_detect_btn = QPushButton("重新检测 (应用新参数)")
        self.re_detect_btn.clicked.connect(self.re_detect)
        self.re_detect_btn.setEnabled(False)
        self.re_detect_btn.setStyleSheet("background-color: #2196F3; color: white; font-size: 12px; padding: 8px;")
        left_layout.addWidget(self.re_detect_btn)
        
        left_layout.addStretch()
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        preview_group = QGroupBox("图片对比")
        preview_layout = QHBoxLayout(preview_group)
        
        template_container = QVBoxLayout()
        template_container.addWidget(QLabel("模板图片"))
        self.template_preview = QLabel()
        self.template_preview.setAlignment(Qt.AlignCenter)
        self.template_preview.setMinimumSize(350, 350)
        self.template_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        template_container.addWidget(self.template_preview)
        preview_layout.addLayout(template_container)
        
        test_container = QVBoxLayout()
        test_container.addWidget(QLabel("待检测图片"))
        self.test_preview = QLabel()
        self.test_preview.setAlignment(Qt.AlignCenter)
        self.test_preview.setMinimumSize(350, 350)
        self.test_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        test_container.addWidget(self.test_preview)
        preview_layout.addLayout(test_container)
        
        annotated_container = QVBoxLayout()
        annotated_container.addWidget(QLabel("缺陷标注"))
        self.annotated_preview = QLabel()
        self.annotated_preview.setAlignment(Qt.AlignCenter)
        self.annotated_preview.setMinimumSize(350, 350)
        self.annotated_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        annotated_container.addWidget(self.annotated_preview)
        preview_layout.addLayout(annotated_container)
        
        right_layout.addWidget(preview_group)
        
        result_group = QGroupBox("检测结果")
        result_layout = QGridLayout(result_group)
        
        result_layout.addWidget(QLabel("质量评分:"), 0, 0)
        self.score_label = QLabel("--")
        self.score_label.setFont(QFont("Arial", 18, QFont.Bold))
        result_layout.addWidget(self.score_label, 0, 1)
        
        result_layout.addWidget(QLabel("质量判定:"), 0, 2)
        self.quality_label = QLabel("--")
        self.quality_label.setFont(QFont("Arial", 18, QFont.Bold))
        result_layout.addWidget(self.quality_label, 0, 3)
        
        result_layout.addWidget(QLabel("缺陷数量:"), 0, 4)
        self.defect_count_label = QLabel("--")
        self.defect_count_label.setFont(QFont("Arial", 14, QFont.Bold))
        result_layout.addWidget(self.defect_count_label, 0, 5)
        
        result_layout.addWidget(QLabel("缺陷比例:"), 1, 0)
        self.defect_label = QLabel("--")
        result_layout.addWidget(self.defect_label, 1, 1)
        
        result_layout.addWidget(QLabel("颜色相似度:"), 1, 2)
        self.color_label = QLabel("--")
        result_layout.addWidget(self.color_label, 1, 3)
        
        result_layout.addWidget(QLabel("边缘缺陷:"), 1, 4)
        self.edge_label = QLabel("--")
        result_layout.addWidget(self.edge_label, 1, 5)
        
        result_layout.addWidget(QLabel("亮度差异:"), 2, 0)
        self.brightness_label = QLabel("--")
        result_layout.addWidget(self.brightness_label, 2, 1)
        
        result_layout.addWidget(QLabel("MSE:"), 2, 2)
        self.mse_label = QLabel("--")
        result_layout.addWidget(self.mse_label, 2, 3)
        
        result_layout.addWidget(QLabel("SSIM:"), 2, 4)
        self.ssim_label = QLabel("--")
        result_layout.addWidget(self.ssim_label, 2, 5)
        
        result_layout.addWidget(QLabel("模板匹配:"), 3, 0)
        self.match_label = QLabel("--")
        result_layout.addWidget(self.match_label, 3, 1)
        
        result_layout.addWidget(QLabel("对齐状态:"), 3, 2)
        self.align_label = QLabel("--")
        result_layout.addWidget(self.align_label, 3, 3)
        
        right_layout.addWidget(result_group)
        
        diff_group = QGroupBox("差异分析图")
        diff_layout = QHBoxLayout(diff_group)
        
        diff_container = QVBoxLayout()
        diff_container.addWidget(QLabel("灰度差异"))
        self.diff_preview = QLabel()
        self.diff_preview.setAlignment(Qt.AlignCenter)
        self.diff_preview.setMinimumSize(280, 280)
        self.diff_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        diff_container.addWidget(self.diff_preview)
        diff_layout.addLayout(diff_container)
        
        defect_container = QVBoxLayout()
        defect_container.addWidget(QLabel("缺陷二值图"))
        self.defect_preview = QLabel()
        self.defect_preview.setAlignment(Qt.AlignCenter)
        self.defect_preview.setMinimumSize(280, 280)
        self.defect_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        defect_container.addWidget(self.defect_preview)
        diff_layout.addLayout(defect_container)
        
        edge_container = QVBoxLayout()
        edge_container.addWidget(QLabel("边缘差异"))
        self.edge_preview = QLabel()
        self.edge_preview.setAlignment(Qt.AlignCenter)
        self.edge_preview.setMinimumSize(280, 280)
        self.edge_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        edge_container.addWidget(self.edge_preview)
        diff_layout.addLayout(edge_container)
        
        right_layout.addWidget(diff_group)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1150])
        
        main_layout.addWidget(splitter)
        
        self.load_default_template()
        
    def load_default_template(self):
        template_path = os.path.join(os.path.dirname(__file__), "img", "template.jpg")
        if os.path.exists(template_path):
            self.set_template(template_path)
            self.load_default_images()
    
    def load_default_images(self):
        img_dir = os.path.join(os.path.dirname(__file__), "img")
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                     and f != "template.jpg"]
            self.image_paths = [os.path.join(img_dir, f) for f in sorted(images)]
            self.update_image_list()
    
    def load_template(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模板图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.set_template(file_path)
    
    def set_template(self, file_path):
        if self.detector.load_template(file_path):
            self.template_label.setText(f"模板: {os.path.basename(file_path)}")
            self.template_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #e8f5e9;")
            
            template = cv_imread(file_path)
            self.display_image(template, self.template_preview)
            
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
            if self.detector.template is not None:
                self.detect_btn.setEnabled(True)
    
    def update_image_list(self):
        self.image_list.clear()
        for path in self.image_paths:
            self.image_list.addItem(os.path.basename(path))
    
    def clear_images(self):
        self.image_paths = []
        self.image_list.clear()
        self.detect_btn.setEnabled(False)
        self.clear_results()
    
    def clear_results(self):
        self.test_preview.clear()
        self.annotated_preview.clear()
        self.diff_preview.clear()
        self.defect_preview.clear()
        self.edge_preview.clear()
        self.score_label.setText("--")
        self.score_label.setStyleSheet("")
        self.quality_label.setText("--")
        self.quality_label.setStyleSheet("")
        self.defect_count_label.setText("--")
        self.defect_label.setText("--")
        self.color_label.setText("--")
        self.edge_label.setText("--")
        self.brightness_label.setText("--")
        self.mse_label.setText("--")
        self.ssim_label.setText("--")
        self.match_label.setText("--")
        self.align_label.setText("--")
        self.align_label.setStyleSheet("")
        self.re_detect_btn.setEnabled(False)
        self.current_result = None
    
    def select_image(self, item):
        index = self.image_list.row(item)
        if 0 <= index < len(self.image_paths):
            image = cv_imread(self.image_paths[index])
            self.display_image(image, self.test_preview)
            self.detect_btn.setEnabled(True)
            self.clear_results()
    
    def on_param_changed(self):
        self.detector.set_param('diff_threshold', self.diff_threshold_spin.value())
        self.detector.set_param('blur_size', self.blur_size_spin.value())
        self.detector.set_param('min_defect_area', self.min_area_spin.value())
        self.detector.set_param('defect_weight', self.defect_weight_spin.value())
        self.detector.set_param('color_weight', self.color_weight_spin.value())
        self.detector.set_param('edge_weight', self.edge_weight_spin.value())
        self.detector.set_param('edge_low', self.edge_low_spin.value())
        self.detector.set_param('edge_high', self.edge_high_spin.value())
        self.detector.set_param('use_ssim', self.ssim_check.isChecked())
        self.detector.set_param('use_color_diff', self.color_diff_check.isChecked())
        self.detector.set_param('scratch_detect', self.scratch_check.isChecked())
        self.detector.set_param('use_template_match', self.template_match_check.isChecked())
        
        if self.current_result is not None:
            self.re_detect_btn.setEnabled(True)
    
    def detect_current(self):
        current_row = self.image_list.currentRow()
        if current_row < 0 and self.image_paths:
            current_row = 0
            self.image_list.setCurrentRow(0)
        
        if current_row < 0 or current_row >= len(self.image_paths):
            QMessageBox.warning(self, "提示", "请先选择一张待检测图片")
            return
        
        if self.detector.template is None:
            QMessageBox.warning(self, "提示", "请先加载模板图片")
            return
        
        self.statusBar().showMessage("正在检测...")
        QApplication.processEvents()
        
        image_path = self.image_paths[current_row]
        result, msg = self.detector.detect(image_path)
        
        if result is None:
            QMessageBox.warning(self, "错误", msg)
            return
        
        self.current_result = result
        self.show_result(result)
        self.statusBar().showMessage(f"检测完成: {os.path.basename(image_path)}")
    
    def re_detect(self):
        if self.current_result is not None:
            current_row = self.image_list.currentRow()
            if current_row >= 0 and current_row < len(self.image_paths):
                image_path = self.image_paths[current_row]
                result, msg = self.detector.detect(image_path)
                if result:
                    self.current_result = result
                    self.show_result(result)
                    self.statusBar().showMessage("重新检测完成")
    
    def show_result(self, result):
        self.display_image(result['resized_image'], self.test_preview)
        self.display_image(result['annotated_image'], self.annotated_preview)
        
        self.score_label.setText(f"{result['score']:.1f}")
        
        quality = result['quality']
        self.quality_label.setText(quality)
        if quality == "合格":
            self.quality_label.setStyleSheet("color: green; font-weight: bold;")
            self.score_label.setStyleSheet("color: green; font-weight: bold;")
        elif quality == "轻微缺陷":
            self.quality_label.setStyleSheet("color: orange; font-weight: bold;")
            self.score_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.quality_label.setStyleSheet("color: red; font-weight: bold;")
            self.score_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.defect_count_label.setText(str(result['defect_count']))
        self.defect_count_label.setStyleSheet("color: red; font-weight: bold;" if result['defect_count'] > 0 else "color: green;")
        
        self.defect_label.setText(f"{result['defect_ratio']*100:.2f}%")
        self.color_label.setText(f"{result['color_similarity']*100:.1f}%")
        self.edge_label.setText(f"{result['edge_defect_ratio']*100:.2f}%")
        self.brightness_label.setText(f"{result['brightness_diff']*100:.1f}%")
        self.mse_label.setText(f"{result['mse']:.1f}")
        
        if 'ssim_score' in result:
            self.ssim_label.setText(f"{result['ssim_score']*100:.1f}%")
        else:
            self.ssim_label.setText("--")
        
        if 'template_match_score' in result:
            self.match_label.setText(f"{result['template_match_score']*100:.1f}%")
        else:
            self.match_label.setText("--")
        
        if 'template_aligned' in result:
            if result['template_aligned']:
                self.align_label.setText("已对齐")
                self.align_label.setStyleSheet("color: green;")
            else:
                self.align_label.setText("未对齐")
                self.align_label.setStyleSheet("color: orange;")
        else:
            self.align_label.setText("--")
        
        self.display_gray_image(result['diff_image'], self.diff_preview)
        self.display_gray_image(result['defect_image'], self.defect_preview)
        self.display_gray_image(result['edge_diff_image'], self.edge_preview)
        
        self.re_detect_btn.setEnabled(True)
    
    def display_image(self, cv_image, label):
        if cv_image is None:
            return
        h, w = cv_image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
    
    def display_gray_image(self, gray_image, label):
        if gray_image is None:
            return
        h, w = gray_image.shape[:2]
        bytes_per_line = w
        q_image = QImage(gray_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PrintQualityChecker()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
