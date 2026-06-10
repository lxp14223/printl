import cv2
import numpy as np
import sys
import os

_dnn_matcher = None


def _get_dnn_matcher():
    global _dnn_matcher
    if _dnn_matcher is None:
        try:
            git_path = os.path.join(os.path.dirname(__file__), "..", "git")
            if git_path not in sys.path:
                sys.path.insert(0, git_path)
            from point_match_dnn import PointMatch

            _dnn_matcher = PointMatch()
            print("深度学习匹配模型加载成功 (SuperPoint+SuperGlue)")
        except Exception as e:
            print(f"深度学习匹配模型加载失败: {e}")
            _dnn_matcher = False
    return _dnn_matcher if _dnn_matcher is not False else None


"""读取中文路径的图片"""


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


"""通过点击区域去除背景"""


def remove_background(image, center, radius=30):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    bg_pixels = image[mask == 255]
    if len(bg_pixels) == 0:
        return image, None

    bg_color = np.mean(bg_pixels, axis=0)

    diff = np.abs(image.astype(np.float32) - bg_color)
    diff_gray = np.max(diff, axis=2)

    _, bg_mask = cv2.threshold(
        diff_gray.astype(np.uint8), 30, 255, cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((5, 5), np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)

    result = image.copy()
    result[bg_mask == 255] = [255, 255, 255]

    return result, bg_mask


"""使用SIFT特征点匹配对齐图像
   将target(扫描图)变换到template(模板)的坐标系，使两者像素对齐"""


def align_images_with_distortion(template, target, min_match_count=10):
    if template is None or target is None:
        return None, None

    template_gray = (
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(template.shape) > 2
        else template
    )
    target_gray = (
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if len(target.shape) > 2 else target
    )

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    if des1 is None or des2 is None:
        return None, None

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print(len(good))
    if len(good) < min_match_count:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        return None, None

    h, w = template_gray.shape[:2]
    aligned = cv2.warpPerspective(target, H, (w, h))

    return aligned, H


"""刚性对齐 - 只允许旋转、平移、均匀缩放，不产生形变"""


def align_images(template, target, min_match_count=10):
    if template is None or target is None:
        return None, None

    template_gray = (
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(template.shape) > 2
        else template
    )
    target_gray = (
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if len(target.shape) > 2 else target
    )

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    if des1 is None or des2 is None:
        return None, None

    # 特征匹配
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < min_match_count:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 用相似变换（4个自由度），替代透视变换（8个自由度）
    M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)

    if M is None or inliers is None:
        return None, None

    # 检查内点数量是否足够
    if np.sum(inliers) < min_match_count:
        return None, None

    h, w = template_gray.shape[:2]

    # 用 warpAffine 替代 warpPerspective
    aligned = cv2.warpAffine(target, M, (w, h))

    # 提取变换参数（调试用）
    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    angle = np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
    tx, ty = M[0, 2], M[1, 2]
    print(f"变换参数: 缩放={scale:.3f}, 旋转={angle:.1f}°, 平移=({tx:.1f}, {ty:.1f})")

    return aligned, M


def align_images_with_dnn(template, target, min_match_count=4):
    """
    使用深度学习特征匹配对齐图像（SuperPoint+SuperGlue）
    比传统SIFT更精确，尤其适合低对比度、小目标场景

    参数:
        template: 模板图像
        target: 待对齐图像
        min_match_count: 最小匹配点数

    返回:
        aligned: 对齐后的图像
        M: 变换矩阵
    """
    if template is None or target is None:
        return None, None

    template_gray = (
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(template.shape) > 2
        else template
    )
    target_gray = (
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if len(target.shape) > 2 else target
    )

    h1, w1 = template_gray.shape[:2]
    h2, w2 = target_gray.shape[:2]
    print(f"DNN对齐: 模板尺寸={w1}x{h1}, 待检测尺寸={w2}x{h2}")

    # ========== 特征匹配 ==========
    try:
        matcher = _get_dnn_matcher()
        if matcher is None:
            print("DNN匹配器不可用，回退到SIFT")
            return align_images(template, target, min_match_count)

        # 使用原始模板和待检测图像进行匹配
        pts0, pts1 = matcher(template_gray, target_gray)

        print(f"DNN找到 {len(pts0)} 个匹配点")

        if len(pts0) < min_match_count:
            print(f"DNN匹配点不足: {len(pts0)} < {min_match_count}，回退到SIFT")
            return align_images(template, target, min_match_count)

        src_pts = pts0.reshape(-1, 1, 2)
        dst_pts = pts1.reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if M is None:
            print("DNN变换矩阵计算失败，回退到SIFT")
            return align_images(template, target, min_match_count)

        inlier_count = np.sum(inliers) if inliers is not None else 0
        if inlier_count < min_match_count:
            print(f"DNN内点数不足: {inlier_count} < {min_match_count}，回退到SIFT")
            return align_images(template, target, min_match_count)

        # 使用原始模板的尺寸进行变换
        h, w = template_gray.shape[:2]
        aligned = cv2.warpAffine(target, M, (w, h))

        scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        angle = np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
        tx, ty = M[0, 2], M[1, 2]
        print(f"DNN对齐成功: {len(pts0)}个匹配点, {inlier_count}个内点")
        print(
            f"变换参数: 缩放={scale:.3f}, 旋转={angle:.1f}°, 平移=({tx:.1f}, {ty:.1f})"
        )

        return aligned, M

    except Exception as e:
        print(f"DNN对齐异常: {e}，回退到SIFT")
        return align_images(template, target, min_match_count)


"""锐化图像 - 解决待检测图片模糊问题"""


def sharpen_image(image, method="high_pass", strength=1.0):
    """
    增强图像清晰度，解决待检测图片模糊的问题

    参数:
        image: 输入图像
        method: 增强方法
            - 'high_pass':  高反差保留（推荐），保留宽范围纹理，不放大噪点
            - 'usm':        USM锐化，效果自然
            - 'laplacian':  拉普拉斯锐化
            - 'combined':   组合锐化（降噪+USM锐化）
        strength: 增强强度 (0.5-2.0)，越大越锐利

    返回:
        sharpened: 增强后的图像
    """
    if image is None:
        return None

    if method == "high_pass":
        # 高反差保留：提取宽范围高频信息叠加回原图
        # 原理: high_pass = original - gaussian_blur(large_sigma)
        #       result  = original + strength * high_pass
        # 使用大sigma保留纹理层级的细节，而非仅保留细边缘
        sigma = 8.0 + strength * 4.0  # sigma范围: 8~16
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        high_pass = cv2.subtract(image, blurred)
        sharpened = cv2.addWeighted(image, 1.0, high_pass, strength, 0)

    elif method == "laplacian":
        # 拉普拉斯锐化
        # 核: [[-1,-1,-1], [-1,8+k,-1], [-1,-1,-1]] / (8+k)
        k = strength * 8
        kernel = np.array(
            [[-1, -1, -1], [-1, 8 + k, -1], [-1, -1, -1]], dtype=np.float32
        ) / (8 + k)
        sharpened = cv2.filter2D(image, -1, kernel)

    elif method == "usm":
        # USM锐化（Unsharp Mask）
        # 原理: sharpened = original + strength * (original - blurred)
        sigma = 1.0 + strength * 0.5
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

    elif method == "combined":
        # 组合锐化：先降噪再锐化
        # 1. 降噪
        if len(image.shape) > 2:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

        # 2. USM锐化
        sigma = 1.0 + strength * 0.5
        blurred = cv2.GaussianBlur(denoised, (0, 0), sigma)
        sharpened = cv2.addWeighted(denoised, 1 + strength, blurred, -strength, 0)

    else:
        sharpened = image

    # 限制范围
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


"""增强图像清晰度 - CLAHE + 锐化"""


def enhance_image_clarity(image, clahe_clip=2.0, sharpen_strength=0.8):
    """
    增强图像清晰度

    步骤:
    1. CLAHE对比度增强
    2. USM锐化

    参数:
        image: 输入图像
        clahe_clip: CLAHE裁剪限制
        sharpen_strength: 锐化强度

    返回:
        enhanced: 增强后的图像
    """
    if image is None:
        return None

    # 1. CLAHE对比度增强
    if len(image.shape) > 2:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

    # 2. USM锐化
    if sharpen_strength > 0:
        enhanced = sharpen_image(enhanced, method="usm", strength=sharpen_strength)

    return enhanced


"""改进的自适应二值化 - 增强对比度，适应不同光照"""


def binarize_image_enhanced(image, block_size=11, c_value=3, use_clahe=True):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0.5)

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )

    return binary


"""自适应二值化（保持兼容）"""


def binarize_image(image, block_size=11, c_value=3):
    return binarize_image_enhanced(image, block_size, c_value, use_clahe=True)


"""形态学处理"""


def morphological_processing(image, kernel_size=3, erode_iter=0, dilate_iter=0):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = image.copy()

    if erode_iter > 0:
        result = cv2.erode(result, kernel, iterations=erode_iter)

    if dilate_iter > 0:
        result = cv2.dilate(result, kernel, iterations=dilate_iter)

    return result
