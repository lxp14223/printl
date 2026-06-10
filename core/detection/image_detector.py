import cv2
import numpy as np
from utils.image_processing import binarize_image, align_images_with_distortion
from utils.connected_components import merge_connected_components, draw_bounding_boxes


class ImageDetector:
    """图文检测模块 - 像素级精度，捕捉每一处差异"""

    def __init__(self):
        self.diff_tolerance = 25
        self.min_area = 5
        self.fill_ratio_threshold = 0.05
        self.morph_kernel = 2
        self.erode_iter = 0
        self.dilate_iter = 0
        self.nms_iou_threshold = 0.3
        self.nms_distance = 20
        self.enable_filter = True
        self.min_color_diff = 10
        self.max_aspect_ratio = 15

    def detect_defects_with_cc(
        self, template, comparison_image, min_area=10, margin=50
    ):
        """
        按连通域分块检测缺陷
        min_area: 从50降到10，检测更小目标
        margin: 从30增加到50，更大上下文
        """
        h, w = template.shape[:2]

        # 转灰度
        if len(template.shape) > 2:
            gray1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(comparison_image, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = template
            gray2 = comparison_image

        # ========== 第2步：找连通域 ==========
        num_labels, labels, stats, centroids = merge_connected_components(
            gray1, min_distance=10, min_area=10
        )

        # draw_bounding_boxes(template, stats, centroids)

        # ========== 第3步：逐个连通域检测 ==========
        all_regions = []  # 所有缺陷区域

        for i in range(0, num_labels):
            try:
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:  # 太小，忽略
                    continue

                # 连通域边界框
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]

                # 扩展边距，确保不越界
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + bw + margin)
                y2 = min(h, y + bh + margin)

                roi1 = template[y1:y2, x1:x2]
                roi2 = comparison_image[y1:y2, x1:x2]

                # 调用检测函数
                roi_regions, roi_diff, roi_diff_mask = self.detect_defects(roi1, roi2)

                if roi_regions is None:
                    continue

                if roi_regions is None or len(roi_regions) == 0:
                    continue

                # ========== 第4步：坐标转换 ==========
                # 格式：{'rect': (x, y, w, h), 'area': float, 'type': str}
                for region in roi_regions:
                    if isinstance(region, dict) and "rect" in region:
                        rx, ry, rw, rh = region["rect"]
                        all_regions.append(
                            {
                                "rect": (rx + x1, ry + y1, rw, rh),
                                "area": region.get("area", 0),
                                "type": region.get("type", "差异"),
                            }
                        )
                    elif isinstance(region, (tuple, list)) and len(region) == 4:
                        rx, ry, rw, rh = region
                        all_regions.append(
                            {
                                "rect": (rx + x1, ry + y1, rw, rh),
                                "area": rw * rh,
                                "type": "差异",
                            }
                        )

                # 处理差异掩膜图（如果需要合并或显示）
                if isinstance(roi_diff_mask, np.ndarray):
                    # 差异掩膜图，可以选择合并到大图中
                    # 例如：将小图的差异掩膜贴到大图的对应位置
                    pass

                # 打印进度
                print(
                    f"连通域 {i}/{num_labels - 1} 检测完成，发现 {len(roi_regions)} 个缺陷"
                )

            except Exception as e:
                print(f"  连通域 {i} 处理失败: {e}")
                continue

        # ========== 第5步：合并重叠的框 ==========
        final_regions = self._merge_overlapping_boxes(all_regions)

        # ========== 第6步：误检过滤 ==========
        final_regions = self.filter_false_positives(
            final_regions, template, comparison_image
        )

        # 这里简化处理，直接对整体做一次差分
        diff = cv2.absdiff(gray1, gray2)
        _, final_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        return final_regions, final_diff

    def detect_defects(self, template, matched_region):
        """
        图文缺陷检测主函数

        功能：对比模板图像和待测图像，检测漏印（missing）和多余污点（extra）两类缺陷

        Args:
            template: 模板图像（标准样张）
            matched_region: 待检测图像（对齐后的扫描件）

        Returns:
            defect_regions: 缺陷区域列表，每个元素包含位置、面积、类型
            significant_diff: 所有缺陷的合并掩膜
            missing: 漏印缺陷掩膜
            extra: 多余缺陷掩膜
        """

        # ==================== 输入验证 ====================
        # 步骤0：检查输入图像是否有效
        if template is None or matched_region is None:
            return [], None, None, None

        # ==================== 步骤1：灰度转换 ====================
        template_gray = (
            cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            if len(template.shape) > 2
            else template
        )
        matched_gray = (
            cv2.cvtColor(matched_region, cv2.COLOR_BGR2GRAY)
            if len(matched_region.shape) > 2
            else matched_region
        )

        # 二次对齐
        matched_gray, homography = align_images_with_distortion(
            template_gray, matched_gray
        )
        if matched_gray is None:
            matched_gray = (
                cv2.cvtColor(matched_region, cv2.COLOR_BGR2GRAY)
                if len(matched_region.shape) > 2
                else matched_region
            )

        # ==================== 步骤2：高斯模糊降噪 ====================
        template_blur = cv2.GaussianBlur(template_gray, (5, 5), 1)
        matched_blur = cv2.GaussianBlur(matched_gray, (5, 5), 1)
        # cv2.imshow("mh:Template", template_blur)
        # cv2.imshow("mh:Matched", matched_blur)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ==================== 步骤3：二值化处理 ====================
        template_binary = binarize_image(template_blur)
        matched_binary = binarize_image(matched_blur)
        # 显示二值化结果
        # cv2.imshow("ezh:Template", template_binary)
        # cv2.imshow("ezh:Matched", matched_binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ==================== 步骤4：形态学预处理 ====================
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        # 4b. 膨胀操作：让线条变粗
        if self.dilate_iter > 0:
            matched_binary = cv2.dilate(
                matched_binary, kernel, iterations=self.dilate_iter
            )
            # cv2.imshow("pz:Matched", matched_binary)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # 4a. 腐蚀操作：让线条变细
        if self.erode_iter > 0:
            matched_binary = cv2.erode(
                matched_binary, kernel, iterations=self.erode_iter
            )
            # cv2.imshow("fs:Matched", matched_binary)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # ==================== 步骤5：前景提取 ====================
        template_fg_raw = cv2.bitwise_not(template_binary)
        matched_fg_raw = cv2.bitwise_not(matched_binary)
        # 显示前景提取
        # cv2.imshow("qjtq:Template", template_fg_raw)
        # cv2.imshow("qjtq:Matched", matched_fg_raw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ==================== 步骤6：形态学闭运算 ====================
        template_fg = cv2.morphologyEx(template_fg_raw, cv2.MORPH_CLOSE, kernel)
        matched_fg = cv2.morphologyEx(matched_fg_raw, cv2.MORPH_CLOSE, kernel)
        # 显示形态学闭运算
        # cv2.imshow("xtx:Template", template_fg)
        # cv2.imshow("xtx:Matched", matched_fg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ==================== 步骤7：有符号差分计算 ====================
        signed_diff = template_blur.astype(np.float32) - matched_blur.astype(np.float32)

        # ==================== 步骤8：缺陷掩膜生成 ====================
        overlay = np.zeros(
            (template_fg.shape[0], template_fg.shape[1], 3), dtype=np.uint8
        )

        # template_fg 前景 → 绿色通道
        overlay[:, :, 1] = template_fg  # G通道
        # matched_fg 前景 → 红色通道
        overlay[:, :, 2] = matched_fg  # R通道
        # 假设要标记的点坐标为 (x, y)
        # cv2.namedWindow('xtx:Overlay (Green=Template, Red=Matched)', cv2.WINDOW_NORMAL)
        # cv2.imshow('xtx:Overlay (Green=Template, Red=Matched)', overlay)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 8. 只检测不同，不区分具体缺陷
        diff_mask = (template_fg != matched_fg).astype(np.uint8) * 255
        # cv2.imshow("diff_mask", diff_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ==================== 步骤9：缺陷掩膜清理 ====================
        diff = self._clean_defect_mask(diff_mask, template_fg, matched_fg)
        # cv2.imshow("diff", diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ==================== 步骤10：提取缺陷区域信息 ====================
        defect_regions = self._extract_defect_regions(diff)

        # 12. 合并重叠的检测框
        defect_regions = self._merge_overlapping_boxes(defect_regions)

        # ==================== 返回结果 ====================
        return defect_regions, diff, diff_mask

    # 缺陷掩膜清理
    @staticmethod
    def _clean_defect_mask(mask, fg1, fg2):
        result = cv2.bitwise_and(mask, fg1)
        result = cv2.bitwise_and(result, cv2.bitwise_not(fg2))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return result

    # 提取缺陷区域信息
    def _extract_defect_regions(self, diff):
        defect_regions = []

        # 创建标注图片（彩色，用于显示）
        diff_color = (
            cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            if len(diff.shape) == 2
            else diff.copy()
        )

        # 处理所有差异（统一用红色框）
        diff_contours, _ = cv2.findContours(
            diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in diff_contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 1.0
                if fill_ratio >= self.fill_ratio_threshold:
                    defect_regions.append(
                        {"rect": (x, y, w, h), "area": area, "type": "差异"}
                    )
                    # 在差异图上画绿色框
                    cv2.rectangle(diff_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        diff_color,
                        "Diff",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )

        # 添加统计信息
        print(f"检测到缺陷总数: {len(defect_regions)}")

        # # 显示标注结果
        # cv2.imshow('Defects (差异)', diff_color)
        # # 等待按键
        # print("\n按任意键继续...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return defect_regions

    # 合并重叠的检测框
    def _merge_overlapping_boxes(self, regions):
        if len(regions) <= 1:
            return regions

        def boxes_overlap(r1, r2):
            x1, y1, w1, h1 = r1["rect"]
            x2, y2, w2, h2 = r2["rect"]

            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)

            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area1 = w1 * h1
                area2 = w2 * h2
                min_area = min(area1, area2)
                if inter / min_area > self.nms_iou_threshold:
                    return True

            cx1 = x1 + w1 / 2
            cy1 = y1 + h1 / 2
            cx2 = x2 + w2 / 2
            cy2 = y2 + h2 / 2
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if dist < self.nms_distance:
                return True

            return False

        if not regions:
            return []

        clusters = []
        visited = [False] * len(regions)

        for i in range(len(regions)):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            queue = [i]

            while queue:
                current = queue.pop(0)
                for j in range(len(regions)):
                    if visited[j]:
                        continue
                    if boxes_overlap(regions[current], regions[j]):
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)

            clusters.append(cluster)

        result = []
        for cluster in clusters:
            if len(cluster) == 1:
                # 单个差异框，修改类型为"差异"
                region = regions[cluster[0]].copy()
                region["type"] = "差异"
                result.append(region)
            else:
                min_x = min_y = float("inf")
                max_x = max_y = 0
                total_area = 0
                for idx in cluster:
                    x, y, w, h = regions[idx]["rect"]
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)
                    total_area += regions[idx]["area"]

                result.append(
                    {
                        "rect": (min_x, min_y, max_x - min_x, max_y - min_y),
                        "area": total_area,
                        "type": "差异",
                    }
                )
        return result

    def set_params(
        self,
        diff_tolerance=30,
        min_area=20,
        fill_ratio_threshold=0.1,
        morph_kernel=3,
        erode_iter=1,
        dilate_iter=0,
        nms_iou_threshold=0.2,
        nms_distance=30,
    ):
        self.diff_tolerance = diff_tolerance
        self.min_area = min_area
        self.fill_ratio_threshold = fill_ratio_threshold
        self.morph_kernel = morph_kernel
        self.erode_iter = erode_iter
        self.dilate_iter = dilate_iter
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_distance = nms_distance

    def filter_false_positives(self, regions, template, matched):
        """
        基于形状特征和颜色差异过滤误检
        减少因对齐误差、噪声导致的误检
        """
        if not self.enable_filter or template is None or matched is None:
            return regions

        filtered = []
        h, w = template.shape[:2]

        for region in regions:
            x, y, rw, rh = region["rect"]
            area = region["area"]

            if rw <= 0 or rh <= 0:
                continue

            aspect_ratio = max(rw, rh) / max(min(rw, rh), 1)
            if aspect_ratio > self.max_aspect_ratio:
                print(f"过滤: 长宽比过大 {aspect_ratio:.1f} > {self.max_aspect_ratio}")
                continue

            extent = area / (rw * rh) if rw * rh > 0 else 0
            if extent < 0.05:
                print(f"过滤: 填充率过低 {extent:.3f} < 0.05")
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + rw)
            y2 = min(h, y + rh)

            if x2 <= x1 or y2 <= y1:
                continue

            roi1 = template[y1:y2, x1:x2]
            roi2 = matched[y1:y2, x1:x2]

            if roi1.size == 0 or roi2.size == 0:
                continue

            if len(roi1.shape) > 2:
                mean1 = np.mean(roi1, axis=(0, 1))
                mean2 = np.mean(roi2, axis=(0, 1))
                color_diff = np.max(np.abs(mean1 - mean2))
            else:
                mean1 = np.mean(roi1)
                mean2 = np.mean(roi2)
                color_diff = abs(mean1 - mean2)

            if color_diff < self.min_color_diff:
                print(f"过滤: 颜色差异过小 {color_diff:.1f} < {self.min_color_diff}")
                continue

            region["color_diff"] = float(color_diff)
            region["aspect_ratio"] = float(aspect_ratio)
            filtered.append(region)

        print(
            f"误检过滤: {len(regions)} -> {len(filtered)} (过滤 {len(regions) - len(filtered)} 个)"
        )
        return filtered

    # ==================== 论文方法增强 ====================

    def compute_local_structure_response(
        self, image, reference_region=None, epsilon=1e-6
    ):
        """
        计算局部结构响应函数（论文公式2）
        R(x,y) = |∇I|² · σ_I² / (σ_I² + ε) · exp[|μ_I - μ_N|² / 2σ_N²]

        作用：增强缺陷区域与正常纹理的可分辨性

        参数:
            image: 输入图像
            reference_region: 参考正常区域（用于计算μ_N和σ_N）
            epsilon: 防止分母为零的小常数

        返回:
            response: 局部结构响应图
        """
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.float32)

        # 计算梯度 |∇I|
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # 计算局部灰度均值和方差
        kernel_size = 15
        mu_I = cv2.boxFilter(gray, -1, (kernel_size, kernel_size))
        sigma_I_sq = cv2.boxFilter((gray - mu_I) ** 2, -1, (kernel_size, kernel_size))

        # 参考区域的统计量
        if reference_region is not None:
            mu_N = np.mean(reference_region)
            sigma_N_sq = np.var(reference_region) + epsilon
        else:
            # 使用全局统计量作为参考
            mu_N = np.mean(gray)
            sigma_N_sq = np.var(gray) + epsilon

        # 计算响应函数
        # 第一项：梯度响应
        grad_response = grad_mag**2

        # 第二项：纹理响应
        texture_response = sigma_I_sq / (sigma_I_sq + epsilon)

        # 第三项：灰度异常响应
        gray_response = np.exp((mu_I - mu_N) ** 2 / (2 * sigma_N_sq))

        # 综合响应
        response = grad_response * texture_response * gray_response

        # 归一化到0-255
        response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
        response = response.astype(np.uint8)

        return response

    def compute_position_weight(self, template, edge_weight=2.0, texture_weight=1.5):
        """
        计算位置权重函数（论文公式3）
        增强图文边界和纹理变化明显区域的贡献

        参数:
            template: 模板图像
            edge_weight: 边缘区域权重
            texture_weight: 纹理变化区域权重

        返回:
            weight_map: 位置权重图
        """
        if len(template.shape) > 2:
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray = template

        h, w = gray.shape
        weight_map = np.ones((h, w), dtype=np.float32)

        # 1. 边缘检测 - 边界区域权重更高
        edges = cv2.Canny(gray, 50, 150)

        # 膨胀边缘，增加影响范围
        kernel_edge = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)

        # 边缘区域权重
        weight_map[edges_dilated > 0] *= edge_weight

        # 2. 纹理变化检测 - 纹理变化明显区域权重更高
        # 计算局部标准差
        kernel_size = 9
        mean_local = cv2.boxFilter(
            gray.astype(np.float32), -1, (kernel_size, kernel_size)
        )
        var_local = cv2.boxFilter(
            (gray.astype(np.float32) - mean_local) ** 2, -1, (kernel_size, kernel_size)
        )
        std_local = np.sqrt(var_local)

        # 纹理变化明显的区域（标准差大）
        std_threshold = np.percentile(std_local, 70)
        texture_mask = std_local > std_threshold

        weight_map[texture_mask] *= texture_weight

        # 平滑权重图
        weight_map = cv2.GaussianBlur(weight_map, (5, 5), 1)

        return weight_map

    def compute_multi_feature_fusion(
        self, template, matched, w_gray=1.0, w_edge=1.0, w_color=1.0
    ):
        """
        三特征融合判别（论文公式4）
        F = w_g·|I_r - I_t| + w_e·|∇I_r - ∇I_t| + w_c·|C_r - C_t|

        参数:
            template: 模板图像
            matched: 待检测图像
            w_gray: 灰度残差权重
            w_edge: 边缘差异权重
            w_color: 颜色差异权重

        返回:
            fusion_map: 融合特征图
            feature_dict: 各特征分量字典
        """
        # 灰度转换
        if len(template.shape) > 2:
            gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_m = cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray_t = template.astype(np.float32)
            gray_m = matched.astype(np.float32)

        # 1. 灰度残差
        gray_diff = np.abs(gray_m - gray_t)
        gray_diff_norm = cv2.normalize(gray_diff, None, 0, 1, cv2.NORM_MINMAX)

        # 2. 边缘差异（梯度）
        # 计算梯度
        grad_t_x = cv2.Sobel(gray_t, cv2.CV_32F, 1, 0, ksize=3)
        grad_t_y = cv2.Sobel(gray_t, cv2.CV_32F, 0, 1, ksize=3)
        grad_t_mag = np.sqrt(grad_t_x**2 + grad_t_y**2)

        grad_m_x = cv2.Sobel(gray_m, cv2.CV_32F, 1, 0, ksize=3)
        grad_m_y = cv2.Sobel(gray_m, cv2.CV_32F, 0, 1, ksize=3)
        grad_m_mag = np.sqrt(grad_m_x**2 + grad_m_y**2)

        edge_diff = np.abs(grad_m_mag - grad_t_mag)
        edge_diff_norm = cv2.normalize(edge_diff, None, 0, 1, cv2.NORM_MINMAX)

        # 3. 颜色差异
        if len(template.shape) > 2 and len(matched.shape) > 2:
            color_diff = np.max(
                np.abs(matched.astype(np.float32) - template.astype(np.float32)), axis=2
            )
            color_diff_norm = cv2.normalize(color_diff, None, 0, 1, cv2.NORM_MINMAX)
        else:
            color_diff_norm = gray_diff_norm

        # 融合
        fusion_map = (
            w_gray * gray_diff_norm
            + w_edge * edge_diff_norm
            + w_color * color_diff_norm
        )
        fusion_map = cv2.normalize(fusion_map, None, 0, 255, cv2.NORM_MINMAX)
        fusion_map = fusion_map.astype(np.uint8)

        feature_dict = {
            "gray_diff": gray_diff.astype(np.uint8),
            "edge_diff": edge_diff.astype(np.uint8),
            "color_diff": (
                color_diff_norm.astype(np.uint8)
                if isinstance(color_diff_norm, np.ndarray)
                else gray_diff.astype(np.uint8)
            ),
            "fusion": fusion_map,
        }

        return fusion_map, feature_dict
