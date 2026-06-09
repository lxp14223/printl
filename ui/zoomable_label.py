from PyQt5.QtWidgets import QScrollArea, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np


class ZoomableLabel(QScrollArea):
    """可缩放的图像查看器，支持鼠标滚轮缩放"""
    
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.original_pixmap = None
        self._img_data = None
        
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2a2a2a;")
        self.setWidget(self.image_label)
        
        self.setMinimumSize(300, 300)
    
    def set_image(self, cv_image):
        """设置OpenCV图像"""
        if cv_image is None:
            return
        
        if len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        
        cv_image = np.ascontiguousarray(cv_image)
        self._img_data = cv_image.data.tobytes()
        
        h, w = cv_image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(self._img_data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.original_pixmap = QPixmap.fromImage(q_image.copy())
        self.zoom_factor = 1.0
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.original_pixmap is None:
            return
        new_w = int(self.original_pixmap.width() * self.zoom_factor)
        new_h = int(self.original_pixmap.height() * self.zoom_factor)
        scaled = self.original_pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.resize(new_w, new_h)
    
    def zoom_in(self):
        """放大图像"""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor *= 1.25
            self.update_display()
    
    def zoom_out(self):
        """缩小图像"""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor /= 1.25
            self.update_display()
    
    def reset_zoom(self):
        """重置缩放"""
        self.zoom_factor = 1.0
        self.update_display()
    
    def wheelEvent(self, event):
        """处理鼠标滚轮事件"""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
