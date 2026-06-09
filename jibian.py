import cv2
import numpy as np
import glob
import os

CHESSBOARD_SIZE = (11, 8)    # 内角点 11列 × 8 行
SQUARE_SIZE = 0.025          # 单格25mm = 0.025米
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_IMGS_DIR = os.path.join(SCRIPT_DIR, "C:\\Users\Lenovo\Desktop\Python\CaptureImage")
SAVE_CALIB_FILE = os.path.join(SCRIPT_DIR, "calibration_data.npz")
# =====================================================================

def calibrate_camera():
    # 1. 构建棋盘格3D世界坐标
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    # 2. 读取所有图片
    img_paths = glob.glob(os.path.join(CALIB_IMGS_DIR, "*.png")) + glob.glob(os.path.join(CALIB_IMGS_DIR, "*.jpg"))
    if not img_paths:
        print(f"错误：文件夹 {CALIB_IMGS_DIR} 中没有找到图片！")
        return False

    print(f"共找到 {len(img_paths)} 张标定图片，开始逐张检测并预览...\n")

    # 3. 亚像素优化终止条件（精度拉满）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001)

    # 4. 逐张检测
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{idx+1}/{len(img_paths)}] 错误：无法读取图片：{os.path.basename(img_path)}")
            continue

        # 转灰度 + 直方图均衡化，增强棋盘格对比度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 关键优化：提升明暗不均图片的识别率

        # 升级角点检测：增加更多优化标志，鲁棒性拉满
        ret, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FILTER_QUADS  # 过滤掉背景误判的四边形
        )

        # 绘制预览图
        draw_img = img.copy()
        if ret:
            # 亚像素角点优化
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            # 绘制角点和连线
            cv2.drawChessboardCorners(draw_img, CHESSBOARD_SIZE, corners_refined, ret)
            print(f"[{idx+1}/{len(img_paths)}] 识别成功：{os.path.basename(img_path)}")
        else:
            print(f"[{idx+1}/{len(img_paths)}] 识别失败：{os.path.basename(img_path)}")

        # 缩放窗口方便查看，按任意键切下一张
        show_img = cv2.resize(draw_img, (960, 720))
        cv2.imshow("Chessboard Detect Preview", show_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # 5. 有效图片校验
    if len(objpoints) < 10:
        print(f"\n错误：有效识别图片仅 {len(objpoints)} 张，至少需要10张，无法标定！")
        print("建议：删除遮挡、模糊、棋盘格太小的图片，重新拍摄合格图片")
        return False

    # 6. 执行标定
    print(f"\n有效图片 {len(objpoints)} 张，正在计算相机内参和畸变系数，请稍候...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # 7. 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    # 8. 保存标定结果
    np.savez(
        SAVE_CALIB_FILE,
        K=K,
        dist=dist,
        chessboard_size=CHESSBOARD_SIZE,
        square_size=SQUARE_SIZE,
        mean_error=mean_error
    )

    # 9. 输出结果

    print(" 相机标定完成")
    print(f"有效标定图片：{len(objpoints)} 张")
    print(f"平均重投影误差：{mean_error:.4f} 像素")
    print("\n内参矩阵 K：")
    print(K)
    print("\n畸变系数 dist（去畸变核心）：")
    print(dist)
    print(f"\n参数已保存至：{SAVE_CALIB_FILE}")
    return True

if __name__ == "__main__":
    calibrate_camera()