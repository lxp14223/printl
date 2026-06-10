import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout, QSlider, QSpinBox, QComboBox, QCheckBox, QDialog,
    QScrollArea, QFrame, QInputDialog
)

from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint


class ColorSelectionDialog(QDialog):
    """颜色选择对话框，用于选择要去除的颜色"""

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择要去除的颜色")
        self.setMinimumSize(600, 400)
        self.selected_colors = []
        self.color_items = []

        layout = QVBoxLayout(self)

        info_label = QLabel("请选择要去除的颜色（可多选）：")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        layout.addWidget(info_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        scroll_layout.setSpacing(10)

        for i, color_info in enumerate(colors):
            color_widget = ColorWidget(color_info, i)
            color_widget.selected.connect(self.on_color_selected)
            self.color_items.append(color_widget)
            row = i // 4
            col = i % 4
            scroll_layout.addWidget(color_widget, row, col)

        scroll_layout.setRowStretch(scroll_layout.rowCount(), 1)
        scroll_layout.setColumnStretch(4, 1)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        button_layout = QHBoxLayout()

        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("取消全选")
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_btn)

        button_layout.addStretch()

        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 20px;")
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px 20px;")
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def on_color_selected(self, color_index, selected):
        color_info = self.color_items[color_index].color_info
        if selected:
            if color_info not in self.selected_colors:
                self.selected_colors.append(color_info)
        else:
            if color_info in self.selected_colors:
                self.selected_colors.remove(color_info)

    def select_all(self):
        for item in self.color_items:
            item.set_selected(True)

    def deselect_all(self):
        for item in self.color_items:
            item.set_selected(False)


class ColorWidget(QFrame):
    """颜色显示和选择组件"""

    selected = pyqtSignal(int, bool)

    def __init__(self, color_info, index, parent=None):
        super().__init__(parent)
        self.color_info = color_info
        self.index = index
        self.is_selected = False

        self.setFixedSize(120, 140)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.color_label = QLabel()
        self.color_label.setFixedSize(100, 80)
        b, g, r = color_info['color']
        self.color_label.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #999;")
        layout.addWidget(self.color_label, alignment=Qt.AlignCenter)

        info_text = f"占比: {color_info['percentage']:.1f}%\n像素: {color_info['count']}"
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(info_label)

        self.update_style()

    def set_selected(self, selected):
        self.is_selected = selected
        self.update_style()
        self.selected.emit(self.index, selected)

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet("QFrame { background-color: #e3f2fd; border: 3px solid #2196F3; }")
        else:
            self.setStyleSheet("QFrame { background-color: white; border: 1px solid #ccc; }")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_selected(not self.is_selected)


class ReplaceColorDialog(QDialog):
    """替换颜色选择对话框"""

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择替换颜色")
        self.setMinimumSize(500, 400)
        self.selected_color = None

        layout = QVBoxLayout(self)

        info_label = QLabel("请选择要替换成的颜色：")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        layout.addWidget(info_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        scroll_layout.setSpacing(10)

        self.color_items = []
        for i, color_info in enumerate(colors):
            color_widget = ColorWidget(color_info, i)
            color_widget.clicked_signal = lambda idx: self.on_color_clicked(idx)
            color_widget.mousePressEvent = lambda event, idx=i: self.on_color_clicked(idx)
            self.color_items.append(color_widget)
            row = i // 4
            col = i % 4
            scroll_layout.addWidget(color_widget, row, col)

        scroll_layout.setRowStretch(scroll_layout.rowCount(), 1)
        scroll_layout.setColumnStretch(4, 1)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("或自定义颜色:"))

        self.custom_color_btn = QPushButton("选择颜色")
        self.custom_color_btn.clicked.connect(self.choose_custom_color)
        custom_layout.addWidget(self.custom_color_btn)

        self.custom_color_label = QLabel()
        self.custom_color_label.setFixedSize(60, 30)
        self.custom_color_label.setStyleSheet("background-color: white; border: 1px solid #999;")
        custom_layout.addWidget(self.custom_color_label)

        custom_layout.addStretch()
        layout.addLayout(custom_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 20px;")
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px 20px;")
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def on_color_clicked(self, index):
        for i, item in enumerate(self.color_items):
            if i == index:
                item.set_selected(True)
                self.selected_color = item.color_info['color']
            else:
                item.set_selected(False)

    def choose_custom_color(self):
        from PyQt5.QtWidgets import QColorDialog
        from PyQt5.QtGui import QColor
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = (color.blue(), color.green(), color.red())
            self.custom_color_label.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid #999;"
            )
            for item in self.color_items:
                item.set_selected(False)


def extract_main_colors(image, n_colors=12):
    """提取图像中的主要颜色（优化版本，稳定版）"""
    if image is None:
        return []

    h, w = image.shape[:2]
    max_size = 300
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        small_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_image = image

    data = small_image.reshape(-1, 3)
    data = np.float32(data)

    sample_size = min(5000, len(data))
    if len(data) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(data), sample_size, replace=False)
        data = data[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 2.0)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)

    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = len(labels)

    colors = []
    for i, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        if percentage > 0.5:
            colors.append({
                'color': tuple(centers[i]),
                'count': int(count),
                'percentage': percentage
            })

    colors.sort(key=lambda x: x['percentage'], reverse=True)

    return colors


def remove_colors_from_image(image, colors_to_remove, tolerance=20, replace_color=None):
    """从图像中去除指定的颜色，使用指定颜色或附近颜色填充"""
    if image is None or not colors_to_remove:
        return image

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for color_info in colors_to_remove:
        b, g, r = color_info['color']

        lower = np.array([max(0, b - tolerance), max(0, g - tolerance), max(0, r - tolerance)])
        upper = np.array([min(255, b + tolerance), min(255, g + tolerance), min(255, r + tolerance)])

        print(f"匹配颜色范围: BGR({b}, {g}, {r}) -> 范围[{lower}, {upper}]")

        color_mask = cv2.inRange(image, lower, upper)
        matched_pixels = np.sum(color_mask > 0)
        print(f"  找到 {matched_pixels} 个匹配像素")

        mask = cv2.bitwise_or(mask, color_mask)

    white_threshold = 240
    white_mask = cv2.inRange(image,
                             np.array([white_threshold, white_threshold, white_threshold]),
                             np.array([255, 255, 255]))
    white_pixels = np.sum(white_mask > 0)
    print(f"检测到 {white_pixels} 个接近白色的像素，将保护这些像素")

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))

    print(f"去除颜色: 找到 {np.sum(mask > 0)} 个像素需要去除（已排除白色）")

    if np.sum(mask > 0) == 0:
        print("警告: 没有找到匹配的颜色像素!")
        return image

    kernel = np.ones((3, 3), np.uint8)
    mask_before_morphology = mask.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    print(f"闭运算后: {np.sum(mask > 0)} 个像素")

    print(f"最终将去除: {np.sum(mask > 0)} 个像素")

    if replace_color is not None:
        result = image.copy()
        result[mask > 0] = replace_color
        print(f"使用指定颜色替换: BGR{replace_color}")
        return result

    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    print("使用inpainting算法自动填充")

    return result


try:
    from core.detection_engine import DetectionEngine
    from utils.image_processing import cv_imread, remove_background
    from ui.clickable_label import ClickableLabel
    from ui.zoom_compare_dialog import ZoomCompareDialog
except ImportError:
    class DetectionEngine:
        def set_image_detector_params(self, **kwargs): pass

        def set_detection_flags(self, **kwargs): pass

        def detect_print_quality(self, template, image):
            return {'image_defects': {'significant_diff': None, 'defect_regions': []}, 'image_alignment': None}


    def cv_imread(path):
        return cv2.imread(path)


    def remove_background(img, point):
        return img, None


    class ClickableLabel:
        def __init__(self, *args): pass


    class ZoomCompareDialog:
        def __init__(self, *args): pass

        def exec_(self): pass

try:
    import fitz

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


def pdf_to_image(pdf_path, page_index=0, max_size=2000):
    """
    将 PDF 页面转换为 OpenCV 图像

    Args:
        pdf_path: PDF 文件路径
        page_index: 页面索引
        max_size: 最大边长限制，超过会自动缩放
    """
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF 未安装，请运行: pip install pymupdf")

    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        page_index = 0

    page = doc[page_index]

    # 获取页面原始尺寸
    rect = page.rect
    orig_w, orig_h = rect.width, rect.height

    # 计算合适的 DPI，确保图像不会太大
    # 假设原始 PDF 72 DPI，计算需要的缩放比例
    if max(orig_w, orig_h) > max_size:
        scale = max_size / max(orig_w, orig_h)
    else:
        scale = 1500 / 72  # 1500 DPI，足够清晰但不会太大

    zoom = 1
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    doc.close()
    return img


class ClickableLabel(QLabel):
    """可点击的标签，用于显示图像"""
    clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cv_image = None
        self._display_pixmap = None
        self._scale_factor = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self.setScaledContents(False)
        self.setMinimumSize(100, 100)

    def set_cv_image(self, cv_img):
        """设置 OpenCV 图像"""
        if cv_img is None:
            self._cv_image = None
            self._display_pixmap = None
            self.clear()
            return

        self._cv_image = cv_img.copy()
        self._update_display()

    def _update_display(self):
        """更新显示"""
        if self._cv_image is None:
            return

        # 转换颜色
        rgb = cv2.cvtColor(self._cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 计算缩放
        label_size = self.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # 计算缩放因子和偏移
        self._scale_factor = scaled_pixmap.width() / pixmap.width()
        self._offset_x = (label_size.width() - scaled_pixmap.width()) // 2
        self._offset_y = (label_size.height() - scaled_pixmap.height()) // 2

        self._display_pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._cv_image is not None:
            self._update_display()

    def get_image_coords(self, label_x, label_y):
        """将标签坐标转换为原始图像坐标"""
        img_x = int((label_x - self._offset_x) / self._scale_factor)
        img_y = int((label_y - self._offset_y) / self._scale_factor)

        if self._cv_image is not None:
            h, w = self._cv_image.shape[:2]
            img_x = max(0, min(img_x, w - 1))
            img_y = max(0, min(img_y, h - 1))

        return img_x, img_y

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._cv_image is not None:
            img_x, img_y = self.get_image_coords(event.pos().x(), event.pos().y())
            self.clicked.emit(img_x, img_y)
        super().mouseReleaseEvent(event)


class ROISelectableLabel(ClickableLabel):
    """支持矩形选区的标签"""
    roi_selected = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2 (原始图像坐标)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.select_mode = False
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()

    def set_select_mode(self, enable):
        """设置选择模式"""
        self.select_mode = enable
        self.drawing = False
        self.setCursor(Qt.CrossCursor if enable else Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if self.select_mode and event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = self.start_point
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.select_mode and self.drawing:
            self.end_point = event.pos()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.select_mode and self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False

            # 计算选区
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 10 and rect.height() > 10:
                # 转换为原始图像坐标
                x1, y1 = self.get_image_coords(rect.left(), rect.top())
                x2, y2 = self.get_image_coords(rect.right(), rect.bottom())
                self.roi_selected.emit(x1, y1, x2, y2)

            self.set_select_mode(False)
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.select_mode and self.drawing:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)
            painter.end()


class PrintQualityDetector(QMainWindow):
    """印刷质量检测仪主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("印刷质量检测仪")
        self.setMinimumSize(1600, 900)
        self.resize(1920, 1080)

        self.detection_engine = DetectionEngine()

        self.template = None
        self.template_path = ""
        self.current_image = None
        self.current_image_path = ""
        self.image_list = []
        self.vis_data = {}
        self.pdf_full_image = None

        self.display_mode = "template"
        self.color_extraction_mode = False
        self.color_extraction_roi = None

        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        toolbar = QHBoxLayout()

        load_template_btn = QPushButton("加载模板")
        load_template_btn.clicked.connect(self.load_template)
        toolbar.addWidget(load_template_btn)

        self.select_color_roi_btn = QPushButton("框选颜色区域")
        self.select_color_roi_btn.clicked.connect(self.start_color_roi_selection)
        self.select_color_roi_btn.setEnabled(False)
        toolbar.addWidget(self.select_color_roi_btn)

        self.extract_color_btn = QPushButton("提取颜色")
        self.extract_color_btn.clicked.connect(self.extract_template_colors)
        self.extract_color_btn.setEnabled(False)
        toolbar.addWidget(self.extract_color_btn)

        self.select_roi_btn = QPushButton("框选模板区域")
        self.select_roi_btn.clicked.connect(self.start_template_selection)
        self.select_roi_btn.setEnabled(False)
        toolbar.addWidget(self.select_roi_btn)

        load_images_btn = QPushButton("加载图像")
        load_images_btn.clicked.connect(self.load_images)
        toolbar.addWidget(load_images_btn)

        detect_btn = QPushButton("检测当前")
        detect_btn.clicked.connect(self.detect_current)
        toolbar.addWidget(detect_btn)

        batch_detect_btn = QPushButton("批量检测")
        batch_detect_btn.clicked.connect(self.batch_detect)
        toolbar.addWidget(batch_detect_btn)

        exit_btn = QPushButton("退出")
        exit_btn.clicked.connect(self.close)
        toolbar.addWidget(exit_btn)

        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # 主体内容区域
        content_layout = QHBoxLayout()

        # 左侧图像列表
        list_group = QGroupBox("图像列表")
        list_layout = QVBoxLayout(list_group)
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.on_image_selected)
        list_layout.addWidget(self.image_list_widget)
        content_layout.addWidget(list_group, 1)

        image_group = QGroupBox("图像显示")
        image_layout = QVBoxLayout(image_group)

        mode_switch_layout = QHBoxLayout()
        self.template_btn = QPushButton("模板图像")
        self.template_btn.setCheckable(True)
        self.template_btn.setChecked(True)
        self.template_btn.clicked.connect(lambda: self.switch_display_mode("template"))
        mode_switch_layout.addWidget(self.template_btn)

        self.current_btn = QPushButton("当前图像")
        self.current_btn.setCheckable(True)
        self.current_btn.clicked.connect(lambda: self.switch_display_mode("current"))
        mode_switch_layout.addWidget(self.current_btn)

        self.diff_btn = QPushButton("差异图像")
        self.diff_btn.setCheckable(True)
        self.diff_btn.clicked.connect(lambda: self.switch_display_mode("diff"))
        mode_switch_layout.addWidget(self.diff_btn)

        mode_switch_layout.addStretch()
        image_layout.addLayout(mode_switch_layout)

        self.main_display_label = ROISelectableLabel()
        self.main_display_label.setAlignment(Qt.AlignCenter)
        self.main_display_label.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        self.main_display_label.clicked.connect(self.on_main_display_clicked)
        self.main_display_label.roi_selected.connect(self.on_template_roi_selected)
        image_layout.addWidget(self.main_display_label, 1)

        self.template_label = ROISelectableLabel()
        self.current_label = ClickableLabel()
        self.diff_label = ClickableLabel()

        view_detail_btn = QPushButton("查看细节")
        view_detail_btn.clicked.connect(self.view_details)
        image_layout.addWidget(view_detail_btn)

        content_layout.addWidget(image_group, 5)

        params_group = QGroupBox("检测参数")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("差异容忍度:"), 0, 0)
        self.diff_tolerance_slider = QSlider(Qt.Horizontal)
        self.diff_tolerance_slider.setRange(5, 100)
        self.diff_tolerance_slider.setValue(15)
        params_layout.addWidget(self.diff_tolerance_slider, 0, 1)
        self.diff_tolerance_value = QLabel("15")
        params_layout.addWidget(self.diff_tolerance_value, 0, 2)
        self.diff_tolerance_slider.valueChanged.connect(lambda val: self.diff_tolerance_value.setText(str(val)))

        params_layout.addWidget(QLabel("最小缺陷面积:"), 1, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 1000)
        self.min_area_spin.setValue(10)
        params_layout.addWidget(self.min_area_spin, 1, 1)

        params_layout.addWidget(QLabel("填充比例阈值:"), 2, 0)
        self.fill_ratio_spin = QSpinBox()
        self.fill_ratio_spin.setRange(1, 100)
        self.fill_ratio_spin.setValue(5)
        params_layout.addWidget(self.fill_ratio_spin, 2, 1)

        params_layout.addWidget(QLabel("形态学内核:"), 3, 0)
        self.morph_kernel_spin = QSpinBox()
        self.morph_kernel_spin.setRange(1, 15)
        self.morph_kernel_spin.setValue(3)
        self.morph_kernel_spin.setSingleStep(2)
        params_layout.addWidget(self.morph_kernel_spin, 3, 1)

        params_layout.addWidget(QLabel("腐蚀迭代:"), 4, 0)
        self.erode_spin = QSpinBox()
        self.erode_spin.setRange(0, 10)
        self.erode_spin.setValue(0)
        params_layout.addWidget(self.erode_spin, 4, 1)

        params_layout.addWidget(QLabel("膨胀迭代:"), 5, 0)
        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 10)
        self.dilate_spin.setValue(0)
        params_layout.addWidget(self.dilate_spin, 5, 1)

        params_layout.addWidget(QLabel("检测模式:"), 6, 0)
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItems(["全功能检测", "仅图文检测", "仅文本检测", "仅条码检测", "仅色彩检测"])
        params_layout.addWidget(self.detection_mode_combo, 6, 1)

        self.align_checkbox = QCheckBox("自动对齐图像")
        self.align_checkbox.setChecked(True)
        params_layout.addWidget(self.align_checkbox, 7, 0, 1, 2)

        apply_params_btn = QPushButton("应用参数")
        apply_params_btn.clicked.connect(self.apply_params)
        params_layout.addWidget(apply_params_btn, 8, 0, 1, 3)

        content_layout.addWidget(params_group, 1)

        main_layout.addLayout(content_layout)

    def switch_display_mode(self, mode):
        """切换显示模式"""
        self.display_mode = mode

        self.template_btn.setChecked(mode == "template")
        self.current_btn.setChecked(mode == "current")
        self.diff_btn.setChecked(mode == "diff")

        if mode == "template":
            self.main_display_label.set_select_mode(False)
            if self.template is not None:
                self.main_display_label.set_cv_image(self.template)
            else:
                self.main_display_label.clear()
        elif mode == "current":
            self.main_display_label.set_select_mode(False)
            if self.current_image is not None:
                self.main_display_label.set_cv_image(self.current_image)
            else:
                self.main_display_label.clear()
        elif mode == "diff":
            self.main_display_label.set_select_mode(False)
            if 'defect_mask' in self.vis_data and self.vis_data['defect_mask'] is not None:
                self.main_display_label.set_cv_image(self.vis_data['defect_mask'])
            else:
                self.main_display_label.clear()

    def on_main_display_clicked(self, x, y):
        """主显示区域点击处理"""
        if self.display_mode == "template":
            self.on_template_clicked(x, y)
        elif self.display_mode == "current":
            self.on_current_clicked(x, y)

    def load_template(self):
        """加载模板（支持 PDF 和图片）"""
        file_filter = "模板文件 (*.jpg *.jpeg *.png *.bmp *.pdf)"
        if not PDF_SUPPORT:
            file_filter = "图像文件 (*.jpg *.jpeg *.png *.bmp)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模板", ".", file_filter
        )

        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            if not PDF_SUPPORT:
                QMessageBox.warning(self, "错误",
                                    "PDF 支持未启用！\n请安装 PyMuPDF: pip install pymupdf")
                return

            try:
                self.pdf_full_image = pdf_to_image(file_path, max_size=2000)
                self.template = self.pdf_full_image.copy()
                self.template_path = file_path
                self.color_extraction_mode = False
                self.color_extraction_roi = None
                self.switch_display_mode("template")
                self.select_color_roi_btn.setEnabled(True)
                self.extract_color_btn.setEnabled(False)
                self.select_roi_btn.setEnabled(True)

                h, w = self.template.shape[:2]
                QMessageBox.information(self, "成功",
                                        f"PDF 已加载!\n尺寸: {w}x{h}\n分辨率: 300 DPI\n\n"
                                        f"操作步骤：\n"
                                        f"1. 点击\"框选颜色区域\"，框选包含要去除颜色的区域\n"
                                        f"2. 点击\"提取颜色\"，选择要去除的颜色\n"
                                        f"3. 点击\"框选模板区域\"，框选最终要用的模板区域")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"PDF 加载失败:\n{str(e)}")
        else:
            self.template = cv_imread(file_path)
            if self.template is not None:
                self.template_path = file_path
                self.pdf_full_image = None
                self.color_extraction_mode = \
                    False
                self.color_extraction_roi = None
                self.switch_display_mode("template")
                self.select_color_roi_btn.setEnabled(True)
                self.extract_color_btn.setEnabled(False)
                self.select_roi_btn.setEnabled(False)
                QMessageBox.information(self, "成功",
                                        f"模板加载成功!\n\n"
                                        f"操作步骤：\n"
                                        f"1. 点击\"框选颜色区域\"，框选包含要去除颜色的区域\n"
                                        f"2. 点击\"提取颜色\"，选择要去除的颜色")
            else:
                QMessageBox.warning(self, "错误", "无法加载模板图像!")

    def start_color_roi_selection(self):
        """启动颜色区域框选"""
        if self.template is None:
            QMessageBox.warning(self, "提示", "请先加载模板!")
            return

        self.color_extraction_mode = True
        self.color_extraction_roi = None
        self.switch_display_mode("template")
        self.main_display_label.set_select_mode(True)

        QMessageBox.information(self, "操作提示",
                                "请在模板图像上框选包含要提取颜色的区域。\n\n"
                                "操作步骤：\n"
                                "1. 按住鼠标左键拖拽，框选包含目标颜色的区域\n"
                                "2. 松开鼠标完成选择\n"
                                "3. 点击\"提取颜色\"按钮，提取该区域的颜色\n"
                                "4. 选择要去除的颜色")

    def extract_template_colors(self):
        """提取已框选区域的颜色"""
        if self.template is None:
            QMessageBox.warning(self, "提示", "请先加载模板!")
            return

        if self.color_extraction_roi is None:
            QMessageBox.warning(self, "提示", "请先点击\"框选颜色区域\"按钮框选要提取颜色的区域!")
            return

        x1, y1, x2, y2 = self.color_extraction_roi
        roi_image = self.template[y1:y2, x1:x2].copy()

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            colors = extract_main_colors(roi_image, n_colors=12)
            QApplication.restoreOverrideCursor()

            if not colors:
                QMessageBox.warning(self, "提示", "无法从选区提取颜色信息!")
                return

            dialog = ColorSelectionDialog(colors, self)
            if dialog.exec_() == QDialog.Accepted and dialog.selected_colors:
                selected_count = len(dialog.selected_colors)

                replace_dialog = ReplaceColorDialog(colors, self)
                if replace_dialog.exec_() == QDialog.Accepted:
                    replace_color = replace_dialog.selected_color

                    if replace_color is None:
                        QMessageBox.warning(self, "提示", "未选择替换颜色，将使用自动填充!")

                    reply = QMessageBox.question(
                        self, "确认去除",
                        f"确定要去除选中的 {selected_count} 种颜色吗？\n"
                        f"这些颜色将在整个模板中被去除，并用选定颜色填充。",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.Yes:
                        tolerance, ok = QInputDialog.getInt(
                            self, "设置颜色容差",
                            "请输入颜色容差值（0-100）：\n"
                            "值越小，颜色匹配越精确\n"
                            "值越大，匹配的颜色范围越广\n"
                            "建议值：15-30",
                            value=20, min=0, max=100
                        )

                        if not ok:
                            tolerance = 20

                        QApplication.setOverrideCursor(Qt.WaitCursor)
                        self.template = remove_colors_from_image(
                            self.template,
                            dialog.selected_colors,
                            tolerance=tolerance,
                            replace_color=replace_color
                        )
                        QApplication.restoreOverrideCursor()

                        self.switch_display_mode("template")

                        QMessageBox.information(
                            self, "成功",
                            f"已去除 {selected_count} 种颜色!\n"
                            f"使用容差值: {tolerance}\n\n"
                            f"如需继续去除其他颜色，请再次点击\"框选颜色区域\"。\n"
                            f"如需框选模板区域，请点击\"框选模板区域\"。"
                        )
                else:
                    QMessageBox.information(self, "提示", "未选择替换颜色，操作已取消。")
            else:
                QMessageBox.information(self, "提示", "未选择要去除的颜色。")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
            error_msg = f"颜色提取过程中发生错误:\n{str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "错误", error_msg)

    def start_template_selection(self):
        """开始框选模板区域"""
        if self.template is None:
            QMessageBox.warning(self, "提示", "请先加载模板!")
            return

        self.color_extraction_mode = False
        self.switch_display_mode("template")
        self.main_display_label.set_select_mode(True)

        QMessageBox.information(self, "操作提示",
                                "请在模板图像上按住鼠标左键拖拽，框选模板区域。\n松开鼠标完成选择。")

    def on_template_roi_selected(self, x1, y1, x2, y2):
        """处理模板选区完成"""
        if self.color_extraction_mode:
            self.color_extraction_mode = False
            self.main_display_label.set_select_mode(False)

            if self.template is None:
                QMessageBox.warning(self, "提示", "模板图像不存在!")
                return

            h, w = self.template.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                QMessageBox.warning(self, "提示", "选区无效!")
                return

            self.color_extraction_roi = (x1, y1, x2, y2)
            self.extract_color_btn.setEnabled(True)

            QMessageBox.information(self, "成功",
                                    f"颜色区域已选择!\n尺寸: {x2 - x1}x{y2 - y1}\n\n"
                                    f"请点击\"提取颜色\"按钮提取该区域的颜色。")
            return

        if self.template is None:
            return

        h, w = self.template.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return

        cropped = self.template[y1:y2, x1:x2].copy()
        self.template = cropped
        self.switch_display_mode("template")

        QMessageBox.information(self, "成功",
                                f"模板区域已选择!\n尺寸: {x2 - x1}x{y2 - y1}")

    def load_images(self):
        """从本地加载离线图片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择待检测图片", ".",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )

        if not file_paths:
            return

        loaded_count = 0
        for file_path in file_paths:
            image = cv_imread(file_path)
            if image is not None:
                filename = os.path.basename(file_path)
                self.image_list.append(file_path)
                self.image_list_widget.addItem(filename)
                loaded_count += 1

        if loaded_count > 0:
            last_path = file_paths[-1]
            self.current_image = cv_imread(last_path)
            self.current_image_path = last_path
            self.switch_display_mode("current")
            QMessageBox.information(self, "成功", f"已加载 {loaded_count} 张图片")
        else:
            QMessageBox.warning(self, "错误", "无法加载选中的图片!")

    def on_image_selected(self, item):
        """选择图像时的处理"""
        index = self.image_list_widget.row(item)
        if 0 <= index < len(self.image_list):
            file_path = self.image_list[index]
            self.current_image = cv_imread(file_path)
            if self.current_image is not None:
                self.current_image_path = file_path
                self.switch_display_mode("current")

    def on_template_clicked(self, x, y):
        """点击模板图像，去除背景"""
        if self.template is not None:
            result, bg_mask = remove_background(self.template, (x, y))
            if result is not None:
                self.template = result
                self.switch_display_mode("template")
                QMessageBox.information(self, "成功", "模板背景已去除!")

    def on_current_clicked(self, x, y):
        """点击当前图像，去除背景"""
        if self.current_image is not None:
            result, bg_mask = remove_background(self.current_image, (x, y))
            if result is not None:
                self.current_image = result
                self.switch_display_mode("current")
                QMessageBox.information(self, "成功", "当前图像背景已去除!")

    def apply_params(self):
        """应用检测参数"""
        diff_tolerance = self.diff_tolerance_slider.value()
        min_area = self.min_area_spin.value()
        fill_ratio_threshold = self.fill_ratio_spin.value() / 100.0
        morph_kernel = self.morph_kernel_spin.value()
        erode_iter = self.erode_spin.value()
        dilate_iter = self.dilate_spin.value()

        self.detection_engine.set_image_detector_params(
            diff_tolerance=diff_tolerance,
            min_area=min_area,
            fill_ratio_threshold=fill_ratio_threshold,
            morph_kernel=morph_kernel,
            erode_iter=erode_iter,
            dilate_iter=dilate_iter
        )

        # 设置检测模式
        mode = self.detection_mode_combo.currentIndex()
        if mode == 0:  # 全功能检测
            self.detection_engine.set_detection_flags(
                detect_image_defects=True,
                detect_text_errors=True,
                detect_barcodes=True,
                detect_color_differences=True,
                align_images=self.align_checkbox.isChecked()
            )
        elif mode == 1:  # 仅图文检测
            self.detection_engine.set_detection_flags(
                detect_image_defects=True,
                detect_text_errors=False,
                detect_barcodes=False,
                detect_color_differences=False,
                align_images=self.align_checkbox.isChecked()
            )
        elif mode == 2:  # 仅文本检测
            self.detection_engine.set_detection_flags(
                detect_image_defects=False,
                detect_text_errors=True,
                detect_barcodes=False,
                detect_color_differences=False,
                align_images=False
            )
        elif mode == 3:  # 仅条码检测
            self.detection_engine.set_detection_flags(
                detect_image_defects=False,
                detect_text_errors=False,
                detect_barcodes=True,
                detect_color_differences=False,
                align_images=False
            )
        elif mode == 4:  # 仅色彩检测
            self.detection_engine.set_detection_flags(
                detect_image_defects=False,
                detect_text_errors=False,
                detect_barcodes=False,
                detect_color_differences=True,
                align_images=self.align_checkbox.isChecked()
            )

        QMessageBox.information(self, "成功", "参数已应用!")

    def detect_current(self):
        """检测当前图像"""
        if self.template is None:
            QMessageBox.warning(self, "提示", "请先加载模板")
            return

        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先选择待检测图像")
            return

        try:
            self.apply_params()

            result = self.detection_engine.detect_print_quality(self.template, self.current_image)

            if 'image_defects' in result and result['image_defects'].get('significant_diff') is not None:
                diff = result['image_defects']['significant_diff']
                diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

                defect_regions = result['image_defects'].get('defect_regions', [])
                for defect in defect_regions:
                    x, y, w, h = defect['rect']
                    cv2.rectangle(diff_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

                self.vis_data['defect_mask'] = diff_color
                self.vis_data['template'] = self.template
                self.vis_data['matched_region'] = self.current_image
                aligned = result['image_alignment']['aligned_image'] if result.get('image_alignment') else None
                if aligned is not None and len(aligned.shape) == 2:
                    aligned = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
                self.vis_data['aligned_region'] = aligned
                self.vis_data['defect_regions'] = defect_regions

                self.switch_display_mode("diff")

                QMessageBox.information(
                    self, "检测完成",
                    f"检测到 {len(defect_regions)} 处差异区域"
                )
            else:
                QMessageBox.information(self, "检测完成", "未检测到明显差异")
        except Exception as e:
            import traceback
            error_msg = f"检测过程中发生错误:\n{str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "错误", error_msg)

    def batch_detect(self):
        """批量检测所有图像"""
        if self.template is None:
            QMessageBox.warning(self, "提示", "请先加载模板")
            return

        if not self.image_list:
            QMessageBox.warning(self, "提示", "请先加载待检测图像")
            return

        # 应用当前参数
        self.apply_params()

        total_defects = 0

        for i, file_path in enumerate(self.image_list):
            image = cv_imread(file_path)
            if image is None:
                continue

            result = self.detection_engine.detect_print_quality(self.template, image)

            if 'image_defects' in result:
                defects = result['image_defects']['defect_regions']
                total_defects += len(defects)

                # 在列表中标记结果
                item = self.image_list_widget.item(i)
                if item:
                    status = f"[{len(defects)}缺陷]"
                    item.setText(f"{os.path.basename(file_path)} {status}")

        QMessageBox.information(
            self, "批量检测完成",
            f"共检测 {len(self.image_list)} 张图像\n" +
            f"发现 {total_defects} 处缺陷"
        )

    def view_details(self):
        print("self.vis_data['template'],", self.vis_data['template'])
        print("self.vis_data['matched_region']", self.vis_data['matched_region'])
        print("self.vis_data.get('aligned_region')", self.vis_data.get('aligned_region'))
        print("self.vis_data.get('defect_mask')", self.vis_data.get('defect_mask'))
        print("self.vis_data.get('defect_regions')", self.vis_data.get('defect_regions'))
        """查看检测细节"""
        if 'template' in self.vis_data and 'matched_region' in self.vis_data:
            dialog = ZoomCompareDialog(
                self.vis_data['template'],
                self.vis_data['matched_region'],
                aligned_region=self.vis_data.get('aligned_region'),
                defect_mask=self.vis_data.get('defect_mask'),
                defect_regions=self.vis_data.get('defect_regions'),
                parent=self
            )
            dialog.exec_()
        else:
            QMessageBox.warning(self, "提示", "请先执行检测")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PrintQualityDetector()
    window.show()
    sys.exit(app.exec_())
