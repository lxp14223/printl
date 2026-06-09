import cv2
import numpy as np

class ColorDetector:
    """色彩检测模块 - 精准色彩测量，还原设计初衷"""
    
    def __init__(self):
        self.color_tolerance = 5  # LAB颜色空间中的容差
        self.sample_size = 10  # 采样区域大小
    
    def measure_color(self, image, points):
        """测量图像中指定点的色彩值
        
        参数:
            image: 输入图像 (RGB)
            points: 采样点列表 [(x1, y1), (x2, y2), ...]
        
        返回:
            color_measurements: 色彩测量结果列表
        """
        if image is None:
            return []
        
        # 转换到LAB颜色空间，更适合人眼感知的色彩比较
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        color_measurements = []
        
        for i, (x, y) in enumerate(points):
            # 确保点在图像范围内
            h, w = image.shape[:2]
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            
            # 采样区域
            half_size = self.sample_size // 2
            x_start = max(0, x - half_size)
            x_end = min(w, x + half_size)
            y_start = max(0, y - half_size)
            y_end = min(h, y + half_size)
            
            # 计算区域内的平均颜色
            sample_region = lab_image[y_start:y_end, x_start:x_end]
            avg_color = np.mean(sample_region, axis=(0, 1))
            
            # 转换回RGB用于显示
            rgb_color = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_LAB2RGB)[0][0]
            
            color_measurements.append({
                'point_id': i,
                'position': (x, y),
                'lab_color': avg_color.tolist(),
                'rgb_color': rgb_color.tolist(),
                'sample_size': (x_end - x_start, y_end - y_start)
            })
        
        return color_measurements
    
    def compare_colors(self, template_colors, scanned_colors):
        """比较模板和扫描图像的色彩差异
        
        参数:
            template_colors: 模板图像的色彩测量结果
            scanned_colors: 扫描图像的色彩测量结果
        
        返回:
            color_differences: 色彩差异列表
        """
        color_differences = []
        
        # 按位置匹配色彩点
        for template_color in template_colors:
            for scanned_color in scanned_colors:
                if template_color['point_id'] == scanned_color['point_id']:
                    # 计算LAB颜色空间中的欧氏距离
                    lab_diff = np.sqrt(np.sum(
                        (np.array(template_color['lab_color']) - np.array(scanned_color['lab_color'])) ** 2
                    ))
                    
                    is_within_tolerance = lab_diff <= self.color_tolerance
                    
                    color_differences.append({
                        'point_id': template_color['point_id'],
                        'position': template_color['position'],
                        'template_lab': template_color['lab_color'],
                        'scanned_lab': scanned_color['lab_color'],
                        'template_rgb': template_color['rgb_color'],
                        'scanned_rgb': scanned_color['rgb_color'],
                        'lab_distance': lab_diff,
                        'is_within_tolerance': is_within_tolerance
                    })
                    
                    break
        
        return color_differences
    
    def analyze_color_uniformity(self, image, region=None):
        """分析图像或区域的色彩均匀性
        
        参数:
            image: 输入图像
            region: 可选的分析区域 (x, y, w, h)
        
        返回:
            uniformity_report: 均匀性报告
        """
        if image is None:
            return None
        
        # 选择分析区域
        if region is not None:
            x, y, w, h = region
            analysis_region = image[max(0, y):min(image.shape[0], y+h), max(0, x):min(image.shape[1], x+w)]
        else:
            analysis_region = image
        
        # 转换到LAB颜色空间
        lab_region = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2LAB)
        
        # 计算颜色分布的标准差
        lab_std = np.std(lab_region, axis=(0, 1))
        
        # 均匀性评分 (0-1，越高越均匀)
        uniformity_score = 1.0 / (1.0 + np.mean(lab_std))
        
        return {
            'region': region,
            'lab_std': lab_std.tolist(),
            'uniformity_score': uniformity_score,
            'is_uniform': uniformity_score > 0.95
        }
    
    def generate_color_report(self, color_differences, uniformity_reports=None):
        """生成色彩检查报告
        
        参数:
            color_differences: 色彩差异列表
            uniformity_reports: 均匀性报告列表
        
        返回:
            report: 色彩检查报告
        """
        total_points = len(color_differences)
        passed_points = sum(1 for diff in color_differences if diff['is_within_tolerance'])
        
        report = {
            'total_points': total_points,
            'passed_points': passed_points,
            'pass_rate': passed_points / total_points if total_points > 0 else 1.0,
            'color_tolerance': self.color_tolerance,
            'differences': color_differences,
            'uniformity_reports': uniformity_reports or []
        }
        
        return report
    
    def set_params(self, color_tolerance=5, sample_size=10):
        """设置色彩检测参数"""
        self.color_tolerance = color_tolerance
        self.sample_size = sample_size