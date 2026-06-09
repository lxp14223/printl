import cv2
import numpy as np
import os

_mtx = None
_dist = None
_map_cache = {}
_init_failed = False


def _find_calibration_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "calibration_data.npz"),
        os.path.join(os.path.dirname(script_dir), "calibration_data.npz"),
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), "calibration_data.npz"),
        "calibration_data.npz",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def init_undistorter(calib_path=None):
    global _mtx, _dist, _init_failed

    if _mtx is not None and _dist is not None:
        return True

    if _init_failed:
        return False

    if calib_path is None:
        calib_path = _find_calibration_file()

    if calib_path is None or not os.path.exists(calib_path):
        print("⚠️ 相机标定文件未找到，跳过去畸变处理")
        _init_failed = True
        return False

    try:
        calib = np.load(calib_path)
        _mtx = calib["K"]
        _dist = calib["dist"]
        print(f"✅ 相机参数加载成功: {calib_path}")
        return True
    except Exception as e:
        print(f"⚠️ 加载相机参数失败: {e}")
        _init_failed = True
        return False


def undistort_image(image):
    global _mtx, _dist, _map_cache, _init_failed

    if not init_undistorter():
        return image

    if _mtx is None or _dist is None:
        return image

    try:
        h, w = image.shape[:2]

        if (h, w) not in _map_cache:
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(_mtx, _dist, (w, h), 1, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(_mtx, _dist, None, new_mtx, (w, h), cv2.CV_32FC1)
            _map_cache[(h, w)] = (mapx, mapy)

        mapx, mapy = _map_cache[(h, w)]
        return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    except Exception as e:
        print(f"⚠️ 去畸变处理失败: {e}")
        return image