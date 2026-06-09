import cv2
import numpy as np
from pathlib import Path


RAW_IMG_DIR = Path(r"D:\Project\PythonProject\internship\background")          # 原始带畸变图片目录
UNDIST_IMG_DIR = Path(r"D:\Project\PythonProject\internship\undistort_background")  # 去畸变后图片保存目录
CALIB_PARAMS_PATH = Path(r"D:\Project\PythonProject\internship\calibration_data.npz")  # 相机参数文件

SUPPORTED_IMG_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

#加载相机内参和畸变系数
def load_camera_params():

    if not CALIB_PARAMS_PATH.exists():
        raise FileNotFoundError(f"相机参数文件不存在：{CALIB_PARAMS_PATH}")

    calib = np.load(CALIB_PARAMS_PATH)
    for key in ["K", "dist"]: # K内参，dist畸变系数
        if key not in calib:
            raise ValueError(f"参数文件缺少：{key}")

    mtx = calib["K"] # mtx相机内参矩阵
    dist = calib["dist"]
    print("✅ 相机参数加载成功")
    return mtx, dist

# 单张图片去畸变核心函数
def undistort_single_image(img_path, mtx, dist):

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    # 计算最优去畸变矩阵
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # 计算映射表
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
    # 执行去畸变
    undist_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return undist_img

def main():

    UNDIST_IMG_DIR.mkdir(parents=True, exist_ok=True)

    mtx, dist = load_camera_params()

    img_list = []
    for fmt in SUPPORTED_IMG_FORMATS:
        img_list.extend(RAW_IMG_DIR.glob(f"*{fmt}"))
        img_list.extend(RAW_IMG_DIR.glob(f"*{fmt.upper()}"))

    img_list = sorted(list(set(img_list)))
    print(f"📂 找到 {len(img_list)} 张图片，开始批量去畸变...")

    success = 0
    for img_path in img_list:
        print(f"正在处理：{img_path.name}")
        undist_img = undistort_single_image(img_path, mtx, dist)
        if undist_img is None:
            print(f"❌ 读取失败：{img_path.name}")
            continue

        # 保存去畸变图片
        save_path = UNDIST_IMG_DIR / img_path.name
        cv2.imwrite(str(save_path), undist_img)
        success += 1

    print(f"\n🎉 全部完成！")
    print(f"成功去畸变：{success}/{len(img_list)} 张")
    print(f"保存路径：{UNDIST_IMG_DIR.absolute()}")

if __name__ == "__main__":
    main()