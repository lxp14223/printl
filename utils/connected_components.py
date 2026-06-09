import cv2
import numpy as np
from utils.image_processing import binarize_image

# 划分连通域
def merge_connected_components(template, min_distance=15, min_area=50):
    binary = binarize_image(template)

    # 反转：把黑色内容变成白色前景
    # 现在：白色(255)=内容，黑色(0)=背景
    binary = cv2.bitwise_not(binary)

    # 加黑边
    binary = cv2.copyMakeBorder(
        binary,
        top=1, bottom=1, left=1, right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    # 找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 2:
        return num_labels, labels, stats, centroids

    # 去掉黑边，调整坐标
    labels = labels[1:-1, 1:-1]

    for i in range(num_labels):
        stats[i, cv2.CC_STAT_LEFT] = max(0, stats[i, cv2.CC_STAT_LEFT] - 1)
        stats[i, cv2.CC_STAT_TOP] = max(0, stats[i, cv2.CC_STAT_TOP] - 1)
        centroids[i][0] = max(0, centroids[i][0] - 1)
        centroids[i][1] = max(0, centroids[i][1] - 1)

    return _merge_close_components(
        labels, stats, centroids, min_distance, min_area
    )

# 合并连通域
def _merge_close_components(labels, stats, centroids, min_distance, min_area):
    num_labels = stats.shape[0]

    if num_labels <= 2:
        return num_labels, labels, stats, centroids

    # 提取连通域信息（跳过背景）
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        components.append({
            'label': i,
            'x1': x,
            'y1': y,
            'x2': x + w,
            'y2': y + h,
            'cx': cx,
            'cy': cy,
            'area': area,
            'mask': (labels == i)
        })

    n = len(components)
    if n <= 1:
        return num_labels, labels, stats, centroids

    # 并查集
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    # 两两比较，判断是否合并
    for i in range(n):
        for j in range(i + 1, n):
            comp1 = components[i]
            comp2 = components[j]

            # 条件1：包含关系
            if _is_inside(comp1, comp2):
                union(i, j)
                continue

            # 条件2：距离近
            if _is_close(comp1, comp2, min_distance):
                union(i, j)
                continue

    # 分组
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    group_list = list(groups.values())

    # 如果没合并任何组，直接返回原始结果
    if len(group_list) == n:
        return num_labels, labels, stats, centroids

    # 创建新的标签图
    new_labels = np.zeros_like(labels)
    new_num = len(group_list) + 1  # +1给背景
    new_stats = np.zeros((new_num, 5), dtype=np.int32)
    new_centroids = np.zeros((new_num, 2), dtype=np.float64)

    for new_id, group in enumerate(group_list, start=1):
        # 合并掩码
        mask = np.zeros_like(labels, dtype=np.uint8)
        for idx in group:
            mask = mask | components[idx]['mask'].astype(np.uint8)

        new_labels[mask > 0] = new_id

        # 计算边界框（所有组件的并集）
        x1 = min(components[idx]['x1'] for idx in group)
        y1 = min(components[idx]['y1'] for idx in group)
        x2 = max(components[idx]['x2'] for idx in group)
        y2 = max(components[idx]['y2'] for idx in group)

        # 计算总面积
        total_area = sum(components[idx]['area'] for idx in group)

        # 计算加权中心
        total_cx = sum(components[idx]['cx'] * components[idx]['area'] for idx in group)
        total_cy = sum(components[idx]['cy'] * components[idx]['area'] for idx in group)
        avg_cx = total_cx / total_area if total_area > 0 else 0
        avg_cy = total_cy / total_area if total_area > 0 else 0

        new_stats[new_id] = [x1, y1, x2 - x1, y2 - y1, total_area]
        new_centroids[new_id] = [avg_cx, avg_cy]

    return new_num, new_labels, new_stats, new_centroids

# 判断两个连通域是否有包含关系
def _is_inside(comp1, comp2):
    # comp1 包含 comp2
    if (comp1['x1'] <= comp2['x1'] and
            comp1['y1'] <= comp2['y1'] and
            comp1['x2'] >= comp2['x2'] and
            comp1['y2'] >= comp2['y2']):
        return True

    # comp2 包含 comp1
    if (comp2['x1'] <= comp1['x1'] and
            comp2['y1'] <= comp1['y1'] and
            comp2['x2'] >= comp1['x2'] and
            comp2['y2'] >= comp1['y2']):
        return True

    return False

# 判断两个连通域是否距离近
def _is_close(comp1, comp2, min_distance):
    x1_1, y1_1, x2_1, y2_1 = comp1['x1'], comp1['y1'], comp1['x2'], comp1['y2']
    x1_2, y1_2, x2_2, y2_2 = comp2['x1'], comp2['y1'], comp2['x2'], comp2['y2']

    # 判断是否重叠
    x_overlap = x1_1 <= x2_2 and x1_2 <= x2_1
    y_overlap = y1_1 <= y2_2 and y1_2 <= y2_1

    if x_overlap and y_overlap:
        # 已经重叠，距离为0，应该合并
        return True

    # 计算X方向距离
    if x_overlap:
        dx = 0
    else:
        dx = min(abs(x1_1 - x2_2), abs(x1_2 - x2_1))

    # 计算Y方向距离
    if y_overlap:
        dy = 0
    else:
        dy = min(abs(y1_1 - y2_2), abs(y1_2 - y2_1))

    # 计算欧几里得距离
    distance = (dx ** 2 + dy ** 2) ** 0.5

    return distance < min_distance

# 在图像上绘制连通域的边界框并显示
def draw_bounding_boxes(image, stats, centroids, margin=20):
    # 确保是彩色图
    if len(image.shape) == 2:
        result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result_img = image.copy()

    # 跳过背景标签
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]

        # 添加边距
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)

        # 画矩形框（绿色）
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 标注ID和面积
        label = f"#{i} {area}px"
        cv2.putText(result_img, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示缺陷数量
    defect_count = len(stats) - 1
    cv2.putText(result_img, f"Defects: {defect_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2)

    # 显示图片，按任意键继续
    cv2.namedWindow('Defect Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Defect Detection', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()