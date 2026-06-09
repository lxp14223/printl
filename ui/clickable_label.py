from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal
import cv2
import numpy as np


class ClickableLabel(QLabel):
    """可点击的图像标签，用于背景去除等交互操作"""
    clicked = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_image = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._img_data = None
    
    def set_cv_image(self, cv_image):
        """设置OpenCV图像"""
        self.cv_image = cv_image
        if cv_image is not None:
            if len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            
            cv_image = np.ascontiguousarray(cv_image)
            self._img_data = cv_image.data.tobytes()
            
            h, w = cv_image.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(self._img_data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.setPixmap(QPixmap.fromImage(q_image.copy()))
    
    def mousePressEvent(self, event: QMouseEvent):
        """处理鼠标点击事件"""
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
