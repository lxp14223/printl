import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QSlider, QSpinBox, QComboBox, QCheckBox, QTabWidget, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


class SmartBinarizationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.gray_image = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("智能二值化 - 前景背景分离")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("加载图片")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        file_layout.addWidget(self.load_btn)
        
        self.image_info_label = QLabel("未加载图片")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        file_layout.addWidget(self.image_info_label)
        
        left_layout.addWidget(file_group)
        
        method_group = QGroupBox("二值化方法")
        method_layout = QVBoxLayout(method_group)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Otsu自动阈值",
            "自适应阈值(均值)",
            "自适应阈值(高斯)",
            "手动阈值",
            "OTSU + 高斯模糊",
            "多阈值分割"
        ])
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        method_layout.addWidget(QLabel("选择方法:"))
        method_layout.addWidget(self.method_combo)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_layout = QHBoxLayout()
        self.threshold_layout.addWidget(QLabel("手动阈值:"))
        self.threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("128")
        self.threshold_layout.addWidget(self.threshold_label)
        self.threshold_container = QWidget()
        self.threshold_container.setLayout(self.threshold_layout)
        method_layout.addWidget(self.threshold_container)
        
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(3, 99)
        self.block_size_spin.setValue(11)
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.valueChanged.connect(self.process_image)
        self.adaptive_layout = QHBoxLayout()
        self.adaptive_layout.addWidget(QLabel("块大小:"))
        self.adaptive_layout.addWidget(self.block_size_spin)
        self.adaptive_container = QWidget()
        self.adaptive_container.setLayout(self.adaptive_layout)
        method_layout.addWidget(self.adaptive_container)
        
        self.c_value_spin = QSpinBox()
        self.c_value_spin.setRange(-50, 50)
        self.c_value_spin.setValue(2)
        self.c_value_spin.valueChanged.connect(self.process_image)
        self.c_layout = QHBoxLayout()
        self.c_layout.addWidget(QLabel("C值:"))
        self.c_layout.addWidget(self.c_value_spin)
        self.c_container = QWidget()
        self.c_container.setLayout(self.c_layout)
        method_layout.addWidget(self.c_container)
        
        left_layout.addWidget(method_group)
        
        preprocess_group = QGroupBox("预处理")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        self.blur_check = QCheckBox("高斯模糊")
        self.blur_check.setChecked(False)
        self.blur_check.stateChanged.connect(self.process_image)
        preprocess_layout.addWidget(self.blur_check)
        
        self.blur_size_spin = QSpinBox()
        self.blur_size_spin.setRange(1, 15)
        self.blur_size_spin.setValue(3)
        self.blur_size_spin.setSingleStep(2)
        self.blur_size_spin.valueChanged.connect(self.process_image)
        blur_size_layout = QHBoxLayout()
        blur_size_layout.addWidget(QLabel("模糊核:"))
        blur_size_layout.addWidget(self.blur_size_spin)
        preprocess_layout.addLayout(blur_size_layout)
        
        self.morph_check = QCheckBox("形态学处理")
        self.morph_check.setChecked(False)
        self.morph_check.stateChanged.connect(self.process_image)
        preprocess_layout.addWidget(self.morph_check)
        
        self.morph_kernel_spin = QSpinBox()
        self.morph_kernel_spin.setRange(1, 15)
        self.morph_kernel_spin.setValue(3)
        self.morph_kernel_spin.setSingleStep(2)
        self.morph_kernel_spin.valueChanged.connect(self.process_image)
        morph_kernel_layout = QHBoxLayout()
        morph_kernel_layout.addWidget(QLabel("核大小:"))
        morph_kernel_layout.addWidget(self.morph_kernel_spin)
        preprocess_layout.addLayout(morph_kernel_layout)
        
        self.erode_spin = QSpinBox()
        self.erode_spin.setRange(0, 10)
        self.erode_spin.setValue(1)
        self.erode_spin.valueChanged.connect(self.process_image)
        erode_layout = QHBoxLayout()
        erode_layout.addWidget(QLabel("腐蚀:"))
        erode_layout.addWidget(self.erode_spin)
        preprocess_layout.addLayout(erode_layout)
        
        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 10)
        self.dilate_spin.setValue(1)
        self.dilate_spin.valueChanged.connect(self.process_image)
        dilate_layout = QHBoxLayout()
        dilate_layout.addWidget(QLabel("膨胀:"))
        dilate_layout.addWidget(self.dilate_spin)
        preprocess_layout.addLayout(dilate_layout)
        
        left_layout.addWidget(preprocess_group)
        
        analysis_group = QGroupBox("分析结果")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.otsu_label = QLabel("Otsu阈值: --")
        analysis_layout.addWidget(self.otsu_label)
        
        self.fg_ratio_label = QLabel("前景比例: --")
        analysis_layout.addWidget(self.fg_ratio_label)
        
        self.bg_ratio_label = QLabel("背景比例: --")
        analysis_layout.addWidget(self.bg_ratio_label)
        
        self.fg_mean_label = QLabel("前景均值: --")
        analysis_layout.addWidget(self.fg_mean_label)
        
        self.bg_mean_label = QLabel("背景均值: --")
        analysis_layout.addWidget(self.bg_mean_label)
        
        left_layout.addWidget(analysis_group)
        
        self.process_btn = QPushButton("处理图片")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 12px;")
        left_layout.addWidget(self.process_btn)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px; padding: 12px;")
        left_layout.addWidget(self.save_btn)
        
        left_layout.addStretch()
        
        main_layout.addWidget(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        tabs = QTabWidget()
        
        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)
        self.original_label = QLabel("原图")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(500, 400)
        self.original_label.setStyleSheet("background-color: #2a2a2a;")
        original_layout.addWidget(self.original_label)
        tabs.addTab(original_tab, "原图")
        
        gray_tab = QWidget()
        gray_layout = QVBoxLayout(gray_tab)
        self.gray_label = QLabel("灰度图")
        self.gray_label.setAlignment(Qt.AlignCenter)
        self.gray_label.setMinimumSize(500, 400)
        self.gray_label.setStyleSheet("background-color: #2a2a2a;")
        gray_layout.addWidget(self.gray_label)
        tabs.addTab(gray_tab, "灰度图")
        
        binary_tab = QWidget()
        binary_layout = QVBoxLayout(binary_tab)
        self.binary_label = QLabel("二值化结果")
        self.binary_label.setAlignment(Qt.AlignCenter)
        self.binary_label.setMinimumSize(500, 400)
        self.binary_label.setStyleSheet("background-color: #2a2a2a;")
        binary_layout.addWidget(self.binary_label)
        tabs.addTab(binary_tab, "二值化结果")
        
        fg_tab = QWidget()
        fg_layout = QVBoxLayout(fg_tab)
        self.fg_label = QLabel("前景")
        self.fg_label.setAlignment(Qt.AlignCenter)
        self.fg_label.setMinimumSize(500, 400)
        self.fg_label.setStyleSheet("background-color: #2a2a2a;")
        fg_layout.addWidget(self.fg_label)
        tabs.addTab(fg_tab, "前景")
        
        bg_tab = QWidget()
        bg_layout = QVBoxLayout(bg_tab)
        self.bg_label = QLabel("背景")
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.setMinimumSize(500, 400)
        self.bg_label.setStyleSheet("background-color: #2a2a2a;")
        bg_layout.addWidget(self.bg_label)
        tabs.addTab(bg_tab, "背景")
        
        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        self.hist_label = QLabel("直方图")
        self.hist_label.setAlignment(Qt.AlignCenter)
        self.hist_label.setMinimumSize(500, 400)
        self.hist_label.setStyleSheet("background-color: white;")
        hist_layout.addWidget(self.hist_label)
        tabs.addTab(hist_tab, "直方图")
        
        right_layout.addWidget(tabs)
        
        main_layout.addWidget(right_panel)
        
        self.on_method_changed(0)
    
    def on_method_changed(self, index):
        self.threshold_container.setVisible(index == 3)
        self.adaptive_container.setVisible(index in [1, 2])
        self.c_container.setVisible(index in [1, 2])
        self.process_image()
    
    def on_threshold_changed(self, value):
        self.threshold_label.setText(str(value))
        self.process_image()
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.image = cv_imread(file_path)
            if self.image is not None:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                h, w = self.image.shape[:2]
                self.image_info_label.setText(f"{os.path.basename(file_path)}\n{w} x {h}")
                self.process_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                self.display_image(self.image, self.original_label)
                self.display_gray(self.gray_image, self.gray_label)
                self.process_image()
            else:
                QMessageBox.warning(self, "错误", "无法加载图片")
    
    def process_image(self):
        if self.gray_image is None:
            return
        
        gray = self.gray_image.copy()
        
        if self.blur_check.isChecked():
            blur_size = self.blur_size_spin.value()
            if blur_size % 2 == 0:
                blur_size += 1
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        method_index = self.method_combo.currentIndex()
        
        if method_index == 0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_thresh = _
        elif method_index == 1:
            block_size = self.block_size_spin.value()
            if block_size % 2 == 0:
                block_size += 1
            c_value = self.c_value_spin.value()
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, block_size, c_value
            )
            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method_index == 2:
            block_size = self.block_size_spin.value()
            if block_size % 2 == 0:
                block_size += 1
            c_value = self.c_value_spin.value()
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c_value
            )
            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method_index == 3:
            thresh = self.threshold_slider.value()
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            otsu_thresh = thresh
        elif method_index == 4:
            blur_size = 5
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_thresh = _
        elif method_index == 5:
            binary = self.multi_threshold(gray)
            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if self.morph_check.isChecked():
            kernel_size = self.morph_kernel_spin.value()
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            erode_iter = self.erode_spin.value()
            dilate_iter = self.dilate_spin.value()
            
            if erode_iter > 0:
                binary = cv2.erode(binary, kernel, iterations=erode_iter)
            if dilate_iter > 0:
                binary = cv2.dilate(binary, kernel, iterations=dilate_iter)
        
        self.binary_result = binary
        
        self.display_gray(binary, self.binary_label)
        
        fg_mask = binary == 0
        bg_mask = binary == 255
        
        fg_image = np.zeros_like(self.image)
        fg_image[fg_mask] = self.image[fg_mask]
        self.display_image(fg_image, self.fg_label)
        
        bg_image = np.zeros_like(self.image)
        bg_image[bg_mask] = self.image[bg_mask]
        self.display_image(bg_image, self.bg_label)
        
        self.show_histogram(gray, otsu_thresh)
        
        self.update_analysis(gray, binary, otsu_thresh)
    
    def multi_threshold(self, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                if hist[i] > np.max(hist) * 0.05:
                    peaks.append(i)
        
        if len(peaks) >= 2:
            peaks.sort()
            thresh1 = peaks[0]
            thresh2 = peaks[-1]
            mid_thresh = (thresh1 + thresh2) // 2
            _, binary = cv2.threshold(blurred, mid_thresh, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def show_histogram(self, gray, otsu_thresh):
        hist = np.zeros((300, 512, 3), dtype=np.uint8)
        hist[:] = 255
        
        hist_data = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_data = hist_data.flatten()
        max_val = np.max(hist_data)
        
        if max_val > 0:
            for i in range(256):
                height = int(hist_data[i] / max_val * 280)
                x1 = i * 2
                x2 = x1 + 1
                cv2.rectangle(hist, (x1, 300 - height), (x2, 300), (100, 100, 100), -1)
        
        x_otsu = int(otsu_thresh * 2)
        cv2.line(hist, (x_otsu, 0), (x_otsu, 300), (0, 0, 255), 2)
        cv2.putText(hist, f"Threshold: {int(otsu_thresh)}", (x_otsu + 5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        self.display_image(hist, self.hist_label)
    
    def update_analysis(self, gray, binary, otsu_thresh):
        self.otsu_label.setText(f"Otsu阈值: {int(otsu_thresh)}")
        
        total_pixels = binary.size
        fg_pixels = np.sum(binary == 0)
        bg_pixels = np.sum(binary == 255)
        
        fg_ratio = fg_pixels / total_pixels * 100
        bg_ratio = bg_pixels / total_pixels * 100
        
        self.fg_ratio_label.setText(f"前景比例: {fg_ratio:.1f}%")
        self.bg_ratio_label.setText(f"背景比例: {bg_ratio:.1f}%")
        
        fg_mask = binary == 0
        bg_mask = binary == 255
        
        if np.any(fg_mask):
            fg_mean = np.mean(gray[fg_mask])
            self.fg_mean_label.setText(f"前景均值: {fg_mean:.1f}")
        else:
            self.fg_mean_label.setText("前景均值: --")
        
        if np.any(bg_mask):
            bg_mean = np.mean(gray[bg_mask])
            self.bg_mean_label.setText(f"背景均值: {bg_mean:.1f}")
        else:
            self.bg_mean_label.setText("背景均值: --")
    
    def display_image(self, cv_image, label):
        if cv_image is None:
            return
        h, w = cv_image.shape[:2]
        if len(cv_image.shape) == 3:
            bytes_per_line = 3 * w
            q_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else:
            bytes_per_line = w
            q_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
    
    def display_gray(self, gray, label):
        if gray is None:
            return
        h, w = gray.shape[:2]
        bytes_per_line = w
        q_image = QImage(gray.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
    
    def save_result(self):
        if self.binary_result is None:
            QMessageBox.warning(self, "提示", "没有可保存的结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "图片文件 (*.png *.jpg *.bmp)"
        )
        if file_path:
            cv2.imencode(os.path.splitext(file_path)[1], self.binary_result)[1].tofile(file_path)
            QMessageBox.information(self, "成功", "结果已保存")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SmartBinarizationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
