import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont


class ImageBinaryWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None  # 存储原始图像
        self.current_display_image = None  # 存储当前显示图像
        self.init_ui()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle('图片二值化工具')
        self.setGeometry(300, 300, 1200, 700)

        # 设置主窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #d3d3d3;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
        """)

        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(300)

        # 图片加载区域
        load_group = QGroupBox("图片操作")
        load_layout = QVBoxLayout()

        self.load_btn = QPushButton("选择图片")
        self.load_btn.clicked.connect(self.load_image)
        load_layout.addWidget(self.load_btn)

        self.file_label = QLabel("未选择图片")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #666; padding: 5px;")
        load_layout.addWidget(self.file_label)

        load_group.setLayout(load_layout)
        control_layout.addWidget(load_group)

        # 二值化控制区域
        binary_group = QGroupBox("二值化参数")
        binary_layout = QVBoxLayout()

        self.threshold_label = QLabel("阈值: 127")
        self.threshold_label.setFont(QFont("Arial", 12))
        binary_layout.addWidget(self.threshold_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(127)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(51)
        self.slider.valueChanged.connect(self.on_threshold_changed)
        binary_layout.addWidget(self.slider)

        # 添加阈值说明
        info_label = QLabel("阈值说明:\n• 值越小，黑色区域越大\n• 值越大，白色区域越大")
        info_label.setStyleSheet("color: #888; font-size: 11px; padding: 5px;")
        binary_layout.addWidget(info_label)

        binary_group.setLayout(binary_layout)
        control_layout.addWidget(binary_group)

        # 图片信息区域
        info_group = QGroupBox("图片信息")
        info_layout = QVBoxLayout()

        self.image_info_label = QLabel("未加载图片")
        self.image_info_label.setStyleSheet("color: #666; padding: 5px;")
        info_layout.addWidget(self.image_info_label)

        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)

        # 添加弹性空间
        control_layout.addStretch()

        # 右侧图片显示区域
        display_panel = QWidget()
        display_layout = QHBoxLayout()
        display_panel.setLayout(display_layout)

        # 原始图片显示
        original_group = QGroupBox("原始图片")
        original_layout = QVBoxLayout()
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setStyleSheet("border: 1px solid #cccccc; background-color: white;")
        self.original_label.setText("请选择图片")
        original_layout.addWidget(self.original_label)
        original_group.setLayout(original_layout)

        # 二值化图片显示
        binary_group_display = QGroupBox("二值化结果")
        binary_layout_display = QVBoxLayout()
        self.binary_label = QLabel()
        self.binary_label.setAlignment(Qt.AlignCenter)
        self.binary_label.setMinimumSize(400, 400)
        self.binary_label.setStyleSheet("border: 1px solid #cccccc; background-color: white;")
        self.binary_label.setText("等待处理")
        binary_layout_display.addWidget(self.binary_label)
        binary_group_display.setLayout(binary_layout_display)

        display_layout.addWidget(original_group)
        display_layout.addWidget(binary_group_display)

        # 将控制面板和显示区域添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_panel, 1)

    def load_image(self):
        """加载图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            # 读取图片
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                self.file_label.setText("图片加载失败！")
                return

            # 显示文件路径
            self.file_label.setText(f"已加载: {file_path.split('/')[-1]}")

            # 显示图片信息
            height, width = self.original_image.shape[:2]
            channels = self.original_image.shape[2] if len(self.original_image.shape) > 2 else 1
            self.image_info_label.setText(f"尺寸: {width} x {height}\n通道数: {channels}")

            # 显示原始图片
            self.display_original_image()

            # 应用当前的二值化阈值
            self.apply_threshold()

    def display_original_image(self):
        """显示原始图片"""
        if self.original_image is not None:
            # 转换颜色空间（OpenCV使用BGR，Qt使用RGB）
            if len(self.original_image.shape) == 3:
                display_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            else:
                display_img = self.original_image

            # 调整图片大小以适应显示区域
            h, w = display_img.shape[:2]
            max_size = 400
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                display_img = cv2.resize(display_img, (new_w, new_h))

            # 转换为QPixmap并显示
            self.display_image(display_img, self.original_label)

    def apply_threshold(self):
        """应用二值化阈值"""
        if self.original_image is None:
            return

        # 获取当前阈值
        threshold_value = self.slider.value()

        # 转换为灰度图
        if len(self.original_image.shape) == 3:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.original_image.copy()

        # 应用二值化
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # 显示二值化结果
        self.display_image(binary_image, self.binary_label, is_binary=True)

        # 更新当前显示图像
        self.current_display_image = binary_image

    def display_image(self, image, label, is_binary=False):
        """显示图片到指定的QLabel"""
        if image is None:
            return

        # 确保图像是连续的
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        # 调整图片大小
        h, w = image.shape[:2]
        max_size = 400
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

        # 转换为QImage
        if len(image.shape) == 2:  # 灰度图
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # 彩色图
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # 转换为QPixmap并设置到label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def on_threshold_changed(self, value):
        """阈值滑块值改变时的处理"""
        # 更新标签显示
        self.threshold_label.setText(f"阈值: {value}")

        # 应用新的阈值
        self.apply_threshold()


def main():
    app = QApplication(sys.argv)
    window = ImageBinaryWidget()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()