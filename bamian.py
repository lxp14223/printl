"""
使用 YOLOv8 进行文档版面分析（Layout Analysis）
支持检测：文本、标题、表格、图像、列表、公式等区域
预训练模型基于 DocLayNet 或 PubLayNet 数据集
"""

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# ------------------------------
# 1. 加载预训练模型
# ------------------------------
# 选项1：通用 YOLOv8 模型（不含版面分析特定类别，需自行训练或微调）
# model = YOLO('yolov8n.pt')

# 选项2：推荐使用已在 DocLayNet 上训练的版面分析模型（需下载权重文件）
# 可从 https://github.com/ppaanngggg/DocLayout-YOLO 或 HuggingFace 获取
# 此处示例使用假定的权重路径，实际使用时请替换为真实路径
MODEL_PATH = 'yolov8n-doclaynet.pt'  # 替换为你的模型路径
model = YOLO(MODEL_PATH)

# ------------------------------
# 2. 定义类别映射（DocLayNet 示例）
# ------------------------------
# 根据实际模型调整类别名称和颜色
CLASS_NAMES = {
    0: 'Caption',        # 图表标题
    1: 'Footnote',       # 脚注
    2: 'Formula',        # 公式
    3: 'List-item',      # 列表项
    4: 'Page-footer',    # 页脚
    5: 'Page-header',    # 页眉
    6: 'Picture',        # 图片
    7: 'Section-header', # 章节标题
    8: 'Table',          # 表格
    9: 'Text',           # 正文
    10: 'Title'          # 文档标题
}

# 为每个类别分配固定颜色（BGR 格式）
COLORS = {
    0: (255, 0, 0),      # 蓝
    1: (0, 255, 0),      # 绿
    2: (0, 0, 255),      # 红
    3: (255, 255, 0),    # 青
    4: (255, 0, 255),    # 紫
    5: (0, 255, 255),    # 黄
    6: (128, 0, 128),    # 深紫
    7: (128, 128, 0),    # 橄榄
    8: (0, 128, 128),    # 墨绿
    9: (128, 0, 0),      # 深蓝
    10: (0, 128, 0)      # 深绿
}

# ------------------------------
# 3. 推理函数（单张图片）
# ------------------------------
def detect_layout(image_path, conf_threshold=0.25, save_result=True):
    """
    对输入图像进行版面分析检测
    :param image_path: 图片路径
    :param conf_threshold: 置信度阈值
    :param save_result: 是否保存结果图片
    :return: 检测结果对象
    """
    # 执行推理
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,          # 不自动保存，我们手动绘制保存
        verbose=False
    )
    return results[0]

# ------------------------------
# 4. 可视化检测结果
# ------------------------------
def draw_boxes(image, result):
    """
    在图像上绘制检测框和标签
    :param image: OpenCV 格式图像 (BGR)
    :param result: YOLO 推理结果对象
    :return: 绘制后的图像
    """
    if result.boxes is None:
        return image

    boxes = result.boxes.xyxy.cpu().numpy()      # 边界框坐标 [x1, y1, x2, y2]
    confs = result.boxes.conf.cpu().numpy()      # 置信度
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # 类别ID

    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        label = CLASS_NAMES.get(cls_id, f'Class {cls_id}')
        color = COLORS.get(cls_id, (0, 255, 0))

        # 画矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        text = f'{label} {conf:.2f}'
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # 画标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1
        )
        # 写文字
        cv2.putText(
            image,
            text,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return image

# ------------------------------
# 5. 主流程示例
# ------------------------------
def main():
    # 输入图像路径
    image_path = 'document_sample.jpg'  # 替换为你的文档图片

    # 读取原始图像（OpenCV 格式）
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"无法读取图像: {image_path}")
        return

    # 执行版面检测
    print("正在进行版面分析...")
    result = detect_layout(image_path, conf_threshold=0.3)

    # 绘制检测结果
    img_with_boxes = draw_boxes(img_bgr.copy(), result)

    # 保存结果
    output_path = 'layout_result.jpg'
    cv2.imwrite(output_path, img_with_boxes)
    print(f"结果已保存至: {output_path}")

    # 使用 matplotlib 显示（兼容中文标签）
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Document Layout Analysis Result')
    plt.show()

    # 可选：打印检测到的区域信息
    print("\n检测到的版面元素：")
    if result.boxes:
        for box, conf, cls_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int)
        ):
            print(f"  - {CLASS_NAMES.get(cls_id, f'Class {cls_id}')}: "
                  f"置信度 {conf:.3f}, 位置 {box.astype(int)}")
    else:
        print("  未检测到任何版面元素。")

if __name__ == '__main__':
    main()