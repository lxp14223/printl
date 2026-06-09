import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def binarize_image(img: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """图像二值化"""
    if method == 'otsu':
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def select_image_file(title="选择图片文件"):
    """使用文件对话框选择图片"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 窗口置顶

    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("所有文件", "*.*")
        ]
    )

    root.destroy()
    return file_path


def load_and_resize_image(file_path, target_size=None):
    """加载并可选调整图像大小"""
    if not file_path:
        return None

    img = cv2.imread(file_path)
    if img is None:
        print(f"错误：无法加载图片 {file_path}")
        return None

    if target_size is not None:
        h, w = img.shape[:2]
        target_h, target_w = target_size

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(img, (new_w, new_h))

    return img


class ImageDefectDetector:
    """图文缺陷检测器（带可视化）"""

    def __init__(self, morph_kernel=3, erode_iter=0, dilate_iter=0, diff_tolerance=30):
        self.morph_kernel = morph_kernel
        self.erode_iter = erode_iter
        self.dilate_iter = dilate_iter
        self.diff_tolerance = diff_tolerance

    def _show_images(self, images_dict, title, figsize=(16, 8)):
        """显示多个图像"""
        n = len(images_dict)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n > 1 else [axes]

        for i, (name, img) in enumerate(images_dict.items()):
            if img is not None:
                if len(img.shape) == 2:
                    axes[i].imshow(img, cmap='gray')
                else:
                    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[i].set_title(name, fontsize=10)
                axes[i].axis('off')

        # 隐藏多余的子图
        for i in range(n, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
        input("按 Enter 继续...")
        plt.close()

    def _clean_defect_mask(self, mask, fg1, fg2, min_area=10):
        """清理缺陷掩膜"""
        if mask is None or np.max(mask) == 0:
            return np.zeros_like(mask) if mask is not None else None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(mask)

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(cleaned, [contour], -1, 255, -1)

        return cleaned

    def _extract_defect_regions(self, missing, extra):
        """提取缺陷区域"""
        regions = []

        for mask, defect_type in [(missing, 'missing'), (extra, 'extra')]:
            if mask is not None and np.max(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        'bbox': (x, y, w, h),
                        'area': cv2.contourArea(contour),
                        'type': defect_type
                    })

        return regions

    def _merge_overlapping_boxes(self, regions, iou_threshold=0.3):
        """合并重叠的检测框"""
        return regions

    def detect_defects_with_visualization(self, template, matched_region):
        """带可视化的缺陷检测"""

        print("\n" + "=" * 60)
        print("开始图文缺陷检测 - 逐步可视化")
        print("=" * 60)

        if template is None or matched_region is None:
            print("错误：模板或待测图片为空")
            return [], None, None, None

        # 步骤1: 原始图像
        print("\n步骤 1/9: 原始输入图像")
        self._show_images(
            {'模板 (Template)': template, '待测图 (Matched)': matched_region},
            '步骤1: 原始输入图像'
        )

        # 步骤2: 灰度转换
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) > 2 else template
        matched_gray = cv2.cvtColor(matched_region, cv2.COLOR_BGR2GRAY) if len(
            matched_region.shape) > 2 else matched_region

        print("\n步骤 2/9: 灰度转换")
        self._show_images(
            {'模板灰度图': template_gray, '待测图灰度图': matched_gray},
            '步骤2: 灰度转换 - 将彩色图转为灰度图'
        )

        # 步骤3: 高斯模糊
        template_blur = cv2.GaussianBlur(template_gray, (5, 5), 1)
        matched_blur = cv2.GaussianBlur(matched_gray, (5, 5), 1)

        print("\n步骤 3/9: 高斯模糊降噪")
        self._show_images(
            {'模板模糊后': template_blur, '待测图模糊后': matched_blur},
            '步骤3: 高斯模糊 (5x5) - 抑制噪点'
        )

        # 步骤4: 二值化
        template_binary = binarize_image(template_blur)
        matched_binary = binarize_image(matched_blur)

        print("\n步骤 4/9: 二值化处理")
        self._show_images(
            {'模板二值图': template_binary, '待测图二值图': matched_binary},
            '步骤4: OTSU二值化 - 分离前景和背景'
        )

        # 步骤5: 形态学处理（腐蚀/膨胀）
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        matched_processed = matched_binary.copy()

        if self.erode_iter > 0:
            matched_processed = cv2.erode(matched_processed, kernel, iterations=self.erode_iter)
            print(f"\n步骤 5a/9: 腐蚀操作 (iterations={self.erode_iter})")
            self._show_images(
                {'模板二值图': template_binary,
                 f'待测图腐蚀后 (iter={self.erode_iter})': matched_processed},
                f'步骤5a: 腐蚀 - 让线条变细，容忍糊版'
            )

        if self.dilate_iter > 0:
            matched_processed = cv2.dilate(matched_processed, kernel, iterations=self.dilate_iter)
            print(f"\n步骤 5b/9: 膨胀操作 (iterations={self.dilate_iter})")
            self._show_images(
                {'模板二值图': template_binary,
                 f'待测图膨胀后 (iter={self.dilate_iter})': matched_processed},
                f'步骤5b: 膨胀 - 让线条变粗，容忍断线'
            )

        matched_binary_final = matched_processed

        # 步骤6: 前景提取（取反）
        template_fg_raw = cv2.bitwise_not(template_binary)
        matched_fg_raw = cv2.bitwise_not(matched_binary_final)

        print("\n步骤 6/9: 前景提取（取反操作）")
        self._show_images(
            {'模板前景 (取反)': template_fg_raw,
             '待测图前景 (取反)': matched_fg_raw},
            '步骤6: 取反 - 文字区域变为白色，便于形态学处理'
        )

        # 步骤7: 形态学闭运算
        template_fg = cv2.morphologyEx(template_fg_raw, cv2.MORPH_CLOSE, kernel)
        matched_fg = cv2.morphologyEx(matched_fg_raw, cv2.MORPH_CLOSE, kernel)

        print("\n步骤 7/9: 形态学闭运算")
        self._show_images(
            {'模板前景闭运算': template_fg,
             '待测图前景闭运算': matched_fg},
            '步骤7: 闭运算 (先膨胀后腐蚀) - 填补文字内部空洞'
        )

        # 步骤8: 有符号差分
        signed_diff = template_blur.astype(np.float32) - matched_blur.astype(np.float32)

        # 可视化差分
        diff_vis = np.clip(signed_diff + 128, 0, 255).astype(np.uint8)
        diff_pos = np.clip(signed_diff, 0, 255).astype(np.uint8)  # 正差分（漏印）
        diff_neg = np.clip(-signed_diff, 0, 255).astype(np.uint8)  # 负差分（多余）

        print("\n步骤 8/9: 有符号差分计算")
        self._show_images(
            {'差分图 (中性灰=0)': diff_vis,
             '正差分 (亮区=漏印倾向)': diff_pos,
             '负差分 (亮区=多余倾向)': diff_neg},
            '步骤8: 有符号差分 - signed_diff = template - matched'
        )

        # 步骤9: 缺陷检测
        missing_mask = ((signed_diff > self.diff_tolerance) &
                        (template_fg > 0) &
                        (matched_fg == 0)).astype(np.uint8) * 255

        extra_mask = ((signed_diff < -self.diff_tolerance) &
                      (matched_fg_raw > 0) &
                      (template_fg_raw == 0)).astype(np.uint8) * 255

        print("\n步骤 9/9: 缺陷掩膜生成")

        # 清理缺陷掩膜
        missing = self._clean_defect_mask(missing_mask, template_fg, matched_fg)
        extra = self._clean_defect_mask(extra_mask, matched_fg_raw, template_fg_raw)

        significant_diff = cv2.bitwise_or(missing, extra)

        # 在原始图像上标注缺陷
        result_img = matched_region.copy() if len(matched_region.shape) == 3 else cv2.cvtColor(matched_region,
                                                                                               cv2.COLOR_GRAY2BGR)

        # 标注漏印（红色）
        if missing is not None:
            contours_missing, _ = cv2.findContours(missing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_missing:
                if cv2.contourArea(contour) > 10:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(result_img, "Missing", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 标注多余（蓝色）
        if extra is not None:
            contours_extra, _ = cv2.findContours(extra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_extra:
                if cv2.contourArea(contour) > 10:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(result_img, "Extra", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        self._show_images(
            {'漏印掩膜 (Missing)': missing,
             '多余掩膜 (Extra)': extra,
             '总差异 (Significant Diff)': significant_diff,
             '检测结果标注': result_img},
            '步骤9: 最终检测结果 - 红色=漏印，蓝色=多余'
        )

        # 提取缺陷区域
        defect_regions = self._extract_defect_regions(missing, extra)
        defect_regions = self._merge_overlapping_boxes(defect_regions)

        # 统计结果
        missing_count = len([r for r in defect_regions if r['type'] == 'missing'])
        extra_count = len([r for r in defect_regions if r['type'] == 'extra'])

        print("\n" + "=" * 60)
        print("检测完成！")
        print(f"发现漏印缺陷: {missing_count} 处")
        print(f"发现多余缺陷: {extra_count} 处")
        print("=" * 60)

        return defect_regions, significant_diff, missing, extra


def main():
    """主函数 - 让用户选择图片进行检测"""

    print("=" * 60)
    print("图文缺陷检测 - 逐步可视化演示")
    print("=" * 60)
    print("\n说明：")
    print("1. 首先选择模板图片（标准样张）")
    print("2. 然后选择待检测图片（扫描件）")
    print("3. 程序将逐步展示检测过程")
    print("4. 每看完一步，按 Enter 键继续\n")

    # 选择模板图片
    print("\n请选择模板图片...")
    template_path = select_image_file("请选择模板图片（标准样张）")

    if not template_path:
        print("未选择模板图片，程序退出")
        return

    print(f"已选择模板: {os.path.basename(template_path)}")

    # 选择待测图片
    print("\n请选择待检测图片...")
    matched_path = select_image_file("请选择待检测图片（扫描件）")

    if not matched_path:
        print("未选择待测图片，程序退出")
        return

    print(f"已选择待测图: {os.path.basename(matched_path)}")

    # 加载图片
    print("\n正在加载图片...")
    template = cv2.imread(template_path)
    matched = cv2.imread(matched_path)

    if template is None:
        print(f"错误：无法加载模板图片 {template_path}")
        return

    if matched is None:
        print(f"错误：无法加载待测图片 {matched_path}")
        return

    # 检查尺寸是否一致
    if template.shape != matched.shape:
        print(f"\n警告：两张图片尺寸不一致！")
        print(f"模板尺寸: {template.shape[1]} x {template.shape[0]}")
        print(f"待测图尺寸: {matched.shape[1]} x {matched.shape[0]}")

        response = input("是否将待测图调整为模板尺寸？(y/n): ")
        if response.lower() == 'y':
            matched = cv2.resize(matched, (template.shape[1], template.shape[0]))
            print("已调整待测图尺寸")
        else:
            print("继续使用原始尺寸（可能导致检测不准确）")

    # 创建检测器
    print("\n" + "=" * 60)
    print("配置检测参数")
    print("=" * 60)

    try:
        morph_kernel = int(input("形态学核大小 (默认3): ") or "3")
        erode_iter = int(input("腐蚀迭代次数 (默认0): ") or "0")
        dilate_iter = int(input("膨胀迭代次数 (默认0): ") or "0")
        diff_tolerance = int(input("差分容忍度 (默认30): ") or "30")
    except ValueError:
        print("输入无效，使用默认参数")
        morph_kernel = 3
        erode_iter = 0
        dilate_iter = 0
        diff_tolerance = 30

    detector = ImageDefectDetector(
        morph_kernel=morph_kernel,
        erode_iter=erode_iter,
        dilate_iter=dilate_iter,
        diff_tolerance=diff_tolerance
    )

    print("\n准备开始检测...")
    input("按 Enter 开始逐步可视化检测...")

    # 运行检测
    defect_regions, significant_diff, missing, extra = \
        detector.detect_defects_with_visualization(template, matched)

    print("\n检测完成！感谢使用。")


if __name__ == "__main__":
    main()