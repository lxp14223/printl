import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


def main():
    # 选择图片文件
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if not file_path:
        print("No file selected")
        return

    # 读取图片
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image")
        return

    print(f"Image loaded: {file_path}")
    print(f"Size: {image.shape[1]}x{image.shape[0]} pixels")

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 显示原始图片
    cv2.imshow('Original Image', image)
    print("\nDisplaying original image. Press any key to continue...")
    cv2.waitKey(0)

    # 自适应二值化
    binary_adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # blockSize - 邻域大小
        2  # C - 常数偏移
    )

    # 显示自适应二值化结果
    cv2.imshow('Adaptive Threshold Binary', binary_adaptive)
    print("Displaying adaptive threshold binary. Press any key to continue...")
    cv2.waitKey(0)

    # 创建腐蚀核
    kernel = np.ones((3, 3), np.uint8)

    # 腐蚀操作
    eroded = cv2.erode(binary_adaptive, kernel, iterations=1)

    # 显示腐蚀结果
    cv2.imshow('Erosion Result', eroded)
    print("Displaying erosion result. Press any key to continue...")
    cv2.waitKey(0)

    # 并排对比显示
    compare = np.hstack([binary_adaptive, eroded])
    cv2.imshow('Binary vs Erosion Comparison', compare)
    print("Displaying side-by-side comparison. Press any key to exit...")
    cv2.waitKey(0)

    # 统计信息
    binary_white = np.sum(binary_adaptive == 255)
    eroded_white = np.sum(eroded == 255)
    print(f"\nProcessing complete!")
    print(f"White pixels in binary: {binary_white:,d}")
    print(f"White pixels after erosion: {eroded_white:,d}")
    print(f"Pixels removed by erosion: {binary_white - eroded_white:,d}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()