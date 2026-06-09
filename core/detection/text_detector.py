import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract 未安装，文本检测功能将被禁用")

class TextDetector:
    """文本检测模块 - 智能文本比对，杜绝文字错误"""
    
    def __init__(self):
        self.text_confidence = 0.8
        self.min_text_size = 10
    
    def detect_text_errors(self, template_text, scanned_text):
        """检测文本错误
        
        参数:
            template_text: 模板文本内容 (列表或字符串)
            scanned_text: 扫描文本内容 (列表或字符串)
        
        返回:
            text_errors: 文本错误列表
        """
        text_errors = []

        # 提取纯文本内容
        def extract_text(text_data):
            if isinstance(text_data, str):
                return [text_data]
            elif isinstance(text_data, list):
                texts = []
                for item in text_data:
                    if isinstance(item, dict):
                        # 从OCR结果字典中提取文本
                        texts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        texts.append(item)
                    else:
                        texts.append(str(item))
                return texts
            else:
                return [str(text_data)]

        template_chars = extract_text(template_text)
        scanned_chars = extract_text(scanned_text)

        # 找出所有可能的文本错误
        max_len = max(len(template_chars), len(scanned_chars))

        for i in range(max_len):
            if i >= len(template_chars):
                text_errors.append({
                    'type': 'extra_text',
                    'position': i,
                    'template_content': '',
                    'scanned_content': scanned_chars[i]
                })
            elif i >= len(scanned_chars):
                text_errors.append({
                    'type': 'missing_text',
                    'position': i,
                    'template_content': template_chars[i],
                    'scanned_content': ''
                })
            else:
                if template_chars[i] != scanned_chars[i]:
                    char_errors = self._compare_chars(template_chars[i], scanned_chars[i])
                    if char_errors:
                        text_errors.extend(char_errors)

        return text_errors
    
    @staticmethod
    def _compare_chars(template_str, scanned_str):
        """字符级比较"""
        errors = []
        max_len = max(len(template_str), len(scanned_str))
        
        for i in range(max_len):
            if i >= len(template_str):
                errors.append({
                    'type': 'extra_character',
                    'position': i,
                    'template_char': '',
                    'scanned_char': scanned_str[i]
                })
            elif i >= len(scanned_str):
                errors.append({
                    'type': 'missing_character',
                    'position': i,
                    'template_char': template_str[i],
                    'scanned_char': ''
                })
            elif template_str[i] != scanned_str[i]:
                errors.append({
                    'type': 'character_mismatch',
                    'position': i,
                    'template_char': template_str[i],
                    'scanned_char': scanned_str[i]
                })
        
        return errors

    @staticmethod
    def _preprocess_for_ocr(image):
        """OCR预处理：提升识别准确率"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 降噪
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # 形态学操作
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def recognize_text(self, image):
        """识别图像中的文本"""
        text_results = []

        if not TESSERACT_AVAILABLE:
            return text_results

        try:
            processed_image = self._preprocess_for_ocr(image)
            custom_config = r'--oem 3 --psm 6'

            data = pytesseract.image_to_data(
                processed_image,
                output_type=pytesseract.Output.DICT,
                config=custom_config,
                lang='chi_sim+eng'
            )

            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    text_info = {
                        'text': data['text'][i].strip(),
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        },
                        'block_num': data['block_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i]
                    }
                    text_results.append(text_info)
        except Exception as e:
            print(f"⚠️ OCR识别失败: {e}")

        return text_results

    def set_params(self, text_confidence=0.8, min_text_size=10):
        """设置文本检测参数"""
        self.text_confidence = text_confidence
        self.min_text_size = min_text_size