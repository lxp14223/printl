from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRectF, pyqtSignal
import cv2
import numpy as np


class ImageViewer(QWidget):
    """自由缩放拖动的图像查看器，支持关联操作"""
    
    zoom_changed = pyqtSignal(float, float, float)
    pan_changed = pyqtSignal(float, float)
    defect_clicked = pyqtSignal(int, int, int, int)
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 20.0
        self.offset = QPoint(0, 0)
        self._dragging = False
        self._last_pos = QPoint(0, 0)
        self._img_data = None
        self.defect_rects = []
        self.show_defects = False
        self.linked = True
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def set_image(self, cv_image):
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
        self.offset = QPoint(0, 0)
        self.fit_to_view()
    
    def set_defect_rects(self, rects):
        self.defect_rects = rects
        self.show_defects = len(rects) > 0
        self.update()
    
    def fit_to_view(self):
        if self.original_pixmap is None:
            return
        vw = self.width()
        vh = self.height()
        iw = self.original_pixmap.width()
        ih = self.original_pixmap.height()
        if iw == 0 or ih == 0:
            return
        scale = min(vw / iw, vh / ih) * 0.9
        self.zoom_factor = scale
        self.offset = QPoint(
            int((vw - iw * scale) / 2),
            int((vh - ih * scale) / 2)
        )
        self.update()
    
    def zoom_at(self, pos, factor):
        if self.original_pixmap is None:
            return
        old_zoom = self.zoom_factor
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor * factor))
        ratio = self.zoom_factor / old_zoom
        cx = (pos.x() - self.offset.x()) * ratio
        cy = (pos.y() - self.offset.y()) * ratio
        self.offset = QPoint(int(pos.x() - cx), int(pos.y() - cy))
        self.update()
        if self.linked:
            self.zoom_changed.emit(self.zoom_factor, pos.x(), pos.y())
    
    def pan_by(self, dx, dy):
        self.offset += QPoint(int(dx), int(dy))
        self.update()
        if self.linked:
            self.pan_changed.emit(dx, dy)
    
    def set_zoom_offset(self, zoom_factor, offset):
        self.zoom_factor = zoom_factor
        self.offset = offset
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(42, 42, 42))
        if self.original_pixmap is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, self.title)
            return
        
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        target_rect = QRectF(
            self.offset.x(),
            self.offset.y(),
            self.original_pixmap.width() * self.zoom_factor,
            self.original_pixmap.height() * self.zoom_factor
        )
        painter.drawPixmap(target_rect, self.original_pixmap, QRectF(self.original_pixmap.rect()))
        
        if self.show_defects and self.defect_rects:
            for i, (x, y, w, h, dtype) in enumerate(self.defect_rects):
                rx = self.offset.x() + x * self.zoom_factor
                ry = self.offset.y() + y * self.zoom_factor
                rw = w * self.zoom_factor
                rh = h * self.zoom_factor
                if dtype == '漏印':
                    pen = QPen(QColor(255, 0, 0), 2)
                else:
                    pen = QPen(QColor(255, 0, 0), 2)
                painter.setPen(pen)
                painter.drawRect(QRectF(rx, ry, rw, rh))
                font = painter.font()
                font.setPixelSize(max(10, int(12 * self.zoom_factor / 1.0)))
                painter.setFont(font)
                label = f"{dtype}#{i+1}"
                painter.drawText(QRectF(rx, ry - 20 * self.zoom_factor, rw, 20 * self.zoom_factor), Qt.AlignCenter, label)
    
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.2
        else:
            factor = 1.0 / 1.2
        self.zoom_at(event.pos(), factor)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            if self.show_defects and self.defect_rects:
                pos = event.pos()
                for i, (x, y, w, h, dtype) in enumerate(self.defect_rects):
                    rx = self.offset.x() + x * self.zoom_factor
                    ry = self.offset.y() + y * self.zoom_factor
                    rw = w * self.zoom_factor
                    rh = h * self.zoom_factor
                    if rx <= pos.x() <= rx + rw and ry <= pos.y() <= ry + rh:
                        self.defect_clicked.emit(x, y, w, h)
                        return
    
    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.pan_by(delta.x(), delta.y())
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap is not None and self.zoom_factor <= 1.0:
            self.fit_to_view()
