import cv2
import numpy as np


class BarcodeDetector:
    """条码检测模块 - 条码合规性检查，确保可读性"""
    
    def __init__(self):
        self.barcode_min_area = 100
        self.quality_threshold = 0.8
    
    def detect_barcodes(self, image):
        """检测图像中的条码
        
        参数:
            image: 输入图像 (RGB或灰度)
        
        返回:
            barcodes: 检测到的条码列表
        """
        if image is None:
            return []
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # 自适应阈值处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        barcodes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.barcode_min_area:
                continue
            
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 条码通常是长条形的
            aspect_ratio = w / h
            if aspect_ratio < 2 or aspect_ratio > 10:
                continue
            
            # 计算条码质量评分
            quality = self._calculate_barcode_quality(gray[y:y+h, x:x+w])
            
            barcodes.append({
                'rect': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'quality': quality,
                'decoded_value': None  # 未来扩展：集成条码解码
            })
        
        return barcodes
    
    def _calculate_barcode_quality(self, barcode_region):
        """计算条码质量评分"""
        if barcode_region.size == 0:
            return 0.0
        
        # 简单的质量评估：黑白像素比例和边缘清晰度
        hist = cv2.calcHist([barcode_region], [0], None, [256], [0, 256])
        black_pixels = hist[:128].sum()
        white_pixels = hist[128:].sum()
        total = black_pixels + white_pixels
        
        if total == 0:
            return 0.0
        
        # 计算黑白比例
        ratio_score = 1.0 - abs(black_pixels - white_pixels) / total
        
        # 计算边缘清晰度
        edges = cv2.Canny(barcode_region, 50, 150)
        edge_density = edges.sum() / 255 / edges.size
        edge_score = min(1.0, edge_density * 2)
        
        return (ratio_score + edge_score) / 2
    
    def decode_barcode(self, barcode_region):
        """解码条码
        
        注意: 此方法需要额外的条码解码库支持
        这里提供基础框架，实际使用时需要安装并配置解码库
        """
        # 未来扩展：集成zbar或其他条码解码库
        return None
    
    def check_compliance(self, barcode_data):
        """检查条码合规性
        
        参数:
            barcode_data: 条码数据字典
        
        返回:
            is_compliant: 是否合规
            issues: 问题列表
        """
        issues = []
        
        if barcode_data['quality'] < self.quality_threshold:
            issues.append('条码质量低，可能难以扫描')
        
        if barcode_data['aspect_ratio'] < 2:
            issues.append('条码宽高比异常')
        
        if barcode_data['area'] < self.barcode_min_area:
            issues.append('条码尺寸过小')
        
        return len(issues) == 0, issues
    
    def set_params(self, barcode_min_area=100, quality_threshold=0.8):
        """设置条码检测参数"""
        self.barcode_min_area = barcode_min_area
        self.quality_threshold = quality_threshold
