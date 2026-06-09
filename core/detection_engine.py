from core.detection.barcode_detector import BarcodeDetector
from core.detection.color_detector import ColorDetector
from core.detection.image_detector import ImageDetector
from core.detection.text_detector import TextDetector
from utils.image_processing import align_images, align_images_with_dnn, sharpen_image, enhance_image_clarity
from .undistort import undistort_image
import cv2
import numpy as np


class DetectionEngine:
    """印刷质量检测引擎 - 整合所有检测模块"""
    
    def __init__(self):
        self.image_detector = ImageDetector()
        self.text_detector = TextDetector()
        self.barcode_detector = BarcodeDetector()
        self.color_detector = ColorDetector()
        
        self.align_images_flag = True
        self.use_dnn_alignment = True
        self.detect_image_defects = True
        self.detect_text_errors = True
        self.detect_barcodes = True
        self.detect_color_differences = True
        self.sharpen_scanned = True  # 是否锐化待检测图片
        self.sharpen_method = 'usm'  # 锐化方法: 'laplacian', 'usm', 'combined'
        self.sharpen_strength = 1.0  # 锐化强度

    def detect_print_quality(self, template, scanned_image, color_sample_points=None):
        """执行完整的印刷质量检测
        
        参数:
            template: 模板图像
            scanned_image: 扫描图像
            color_sample_points: 色彩采样点列表
        
        返回:
            detection_result: 检测结果
        """
        detection_result = {
            'image_alignment': None,
            'image_defects': [],
            'text_errors': [],
            'barcodes': [],
            'color_differences': [],
            'color_report': None
        }

        # 先对图像去畸变
        scanned_image = undistort_image(scanned_image)
        
        # 锐化待检测图片，解决模糊问题
        if self.sharpen_scanned:
            print(f"锐化待检测图片 (方法={self.sharpen_method}, 强度={self.sharpen_strength})...")
            scanned_image = sharpen_image(scanned_image, method=self.sharpen_method, strength=self.sharpen_strength)

        # 1. 图像对齐
        if self.align_images_flag:
            if self.use_dnn_alignment:
                print("使用深度学习对齐 (SuperPoint+SuperGlue)...")
                aligned_image, homography = align_images_with_dnn(template, scanned_image)
                
                if aligned_image is None:
                    print("深度学习对齐失败，回退到传统SIFT对齐")
                    aligned_image, homography = align_images(template, scanned_image)
            else:
                print("使用传统SIFT对齐...")
                aligned_image, homography = align_images(template, scanned_image)
            
            if aligned_image is not None:
                detection_result['image_alignment'] = {
                    'aligned_image': aligned_image,
                    'homography_matrix': homography
                }
                comparison_image = aligned_image
            else:
                print("警告: 图像对齐失败，使用原始图像")
                comparison_image = scanned_image
        else:
            comparison_image = scanned_image

        # 2. 图文缺陷检测
        if self.detect_image_defects:
            defect_regions, significant_diff = \
                self.image_detector.detect_defects_with_cc(
                    template, comparison_image,
                    min_area=10,
                    margin=50
                )

            detection_result['image_defects'] = {
                'defect_regions': defect_regions,
                'significant_diff': significant_diff,
            }

        # 3. 文本错误检测
        if self.detect_text_errors:
            template_text = self.text_detector.recognize_text(template)
            scanned_text = self.text_detector.recognize_text(comparison_image)
            
            text_errors = self.text_detector.detect_text_errors(template_text, scanned_text)
            detection_result['text_errors'] = text_errors
            print("text_errors", text_errors)
        
        # 4. 条码检测
        if self.detect_barcodes:
            barcodes = self.barcode_detector.detect_barcodes(comparison_image)
            
            # 检查条码合规性
            for barcode in barcodes:
                is_compliant, issues = self.barcode_detector.check_compliance(barcode)
                barcode['is_compliant'] = is_compliant
                barcode['compliance_issues'] = issues
            
            detection_result['barcodes'] = barcodes
        
        # 5. 色彩差异检测
        if self.detect_color_differences and color_sample_points:
            template_colors = self.color_detector.measure_color(template, color_sample_points)
            scanned_colors = self.color_detector.measure_color(comparison_image, color_sample_points)
            
            color_differences = self.color_detector.compare_colors(template_colors, scanned_colors)
            color_report = self.color_detector.generate_color_report(color_differences)
            
            detection_result['color_differences'] = color_differences
            detection_result['color_report'] = color_report

        return detection_result
    
    def set_sharpen_params(self, enable=True, method='usm', strength=1.0):
        """设置锐化参数
        
        参数:
            enable: 是否启用锐化
            method: 锐化方法 ('laplacian', 'usm', 'combined')
            strength: 锐化强度 (0.5-2.0)
        """
        self.sharpen_scanned = enable
        self.sharpen_method = method
        self.sharpen_strength = strength
    
    def set_image_detector_params(self, **params):
        """设置图文检测参数"""
        self.image_detector.set_params(**params)
    
    def set_text_detector_params(self, **params):
        """设置文本检测参数"""
        self.text_detector.set_params(**params)
    
    def set_barcode_detector_params(self, **params):
        """设置条码检测参数"""
        self.barcode_detector.set_params(**params)
    
    def set_color_detector_params(self, **params):
        """设置色彩检测参数"""
        self.color_detector.set_params(**params)
    
    def set_detection_flags(self, 
                          detect_image_defects=True,
                          detect_text_errors=True,
                          detect_barcodes=True,
                          detect_color_differences=True,
                          align_images=True,
                          use_dnn_alignment=True):
        """设置检测开关"""
        self.detect_image_defects = detect_image_defects
        self.detect_text_errors = detect_text_errors
        self.detect_barcodes = detect_barcodes
        self.detect_color_differences = detect_color_differences
        self.align_images_flag = align_images
        self.use_dnn_alignment = use_dnn_alignment
