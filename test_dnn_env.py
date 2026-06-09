# -*- coding: utf-8 -*-
"""
测试深度学习对齐环境是否就绪
运行此脚本检查所有依赖是否正确安装
"""
import sys
import os

print("=" * 60)
print("深度学习对齐环境检测")
print("=" * 60)

# 1. 检测PyTorch
print("\n[1] 检测 PyTorch...")
try:
    import torch
    print(f"    ✓ PyTorch 版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"    ✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"    ✓ CUDA 版本: {torch.version.cuda}")
    else:
        print("    ⚠ CUDA 不可用，将使用CPU模式（速度较慢）")
except ImportError as e:
    print(f"    ✗ PyTorch 未安装: {e}")
    print("    请运行: pip install torch torchvision")
    sys.exit(1)

# 2. 检测OpenCV
print("\n[2] 检测 OpenCV...")
try:
    import cv2
    print(f"    ✓ OpenCV 版本: {cv2.__version__}")
except ImportError as e:
    print(f"    ✗ OpenCV 未安装: {e}")
    sys.exit(1)

# 3. 检测NumPy
print("\n[3] 检测 NumPy...")
try:
    import numpy as np
    print(f"    ✓ NumPy 版本: {np.__version__}")
except ImportError as e:
    print(f"    ✗ NumPy 未安装: {e}")
    sys.exit(1)

# 4. 检测模型文件
print("\n[4] 检测模型文件...")
base_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(base_dir, "git", "models", "weights")

superpoint_path = os.path.join(weights_dir, "superpoint_v1.pth")
superglue_path = os.path.join(weights_dir, "superglue_outdoor.pth")

if os.path.exists(superpoint_path):
    size_mb = os.path.getsize(superpoint_path) / 1024 / 1024
    print(f"    ✓ SuperPoint 模型: {superpoint_path} ({size_mb:.1f}MB)")
else:
    print(f"    ✗ SuperPoint 模型不存在: {superpoint_path}")
    sys.exit(1)

if os.path.exists(superglue_path):
    size_mb = os.path.getsize(superglue_path) / 1024 / 1024
    print(f"    ✓ SuperGlue 模型: {superglue_path} ({size_mb:.1f}MB)")
else:
    print(f"    ✗ SuperGlue 模型不存在: {superglue_path}")
    sys.exit(1)

# 5. 测试加载模型
print("\n[5] 测试加载模型...")
try:
    sys.path.insert(0, os.path.join(base_dir, "git"))
    from point_match_dnn import PointMatch
    
    print("    正在初始化模型...")
    matcher = PointMatch()
    print("    ✓ 模型加载成功!")
except Exception as e:
    print(f"    ✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试匹配功能
print("\n[6] 测试匹配功能...")
try:
    img1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    pts0, pts1 = matcher(img1, img2)
    print(f"    ✓ 匹配测试成功! 找到 {len(pts0)} 个匹配点")
except Exception as e:
    print(f"    ✗ 匹配测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有检测通过！深度学习对齐环境已就绪")
print("=" * 60)
print("\n下一步：")
print("  1. 运行 print_quality_detector.py")
print("  2. 在参数面板勾选 '使用深度学习对齐'")
print("  3. 加载模板和待检测图像进行检测")
