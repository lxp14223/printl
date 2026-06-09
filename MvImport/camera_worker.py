# -*- coding: utf-8 -*-
import sys
import os
import threading
import time
import numpy as np
import cv2

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QPushButton

sys.path.append(os.getenv('MVCAM_COMMON_RUNENV') + "/Samples/Python/MvImport")
from MvCameraControl_class import *
from CameraParams_header import *


class CameraWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray, float)
    error_signal = pyqtSignal(str)
    connected_signal = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.cam = None
        self.is_grabbing = False
        self.exit_flag = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.trigger_time = None
        self.trigger_lock = threading.Lock()
        
    def enum_devices(self):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                      | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            return None
        
        if deviceList.nDeviceNum == 0:
            return None

        devices = []
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                strModeName = self._get_str(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                devices.append({
                    "index": i,
                    "name": strModeName,
                    "info": "IP: %d.%d.%d.%d" % (nip1, nip2, nip3, nip4)
                })
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                strModeName = self._get_str(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                strSerialNumber = self._get_str(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
                devices.append({
                    "index": i,
                    "name": strModeName,
                    "info": "S/N: %s" % strSerialNumber
                })
        
        return devices, deviceList

    def _get_str(self, char_array):
        result = ""
        for per in char_array:
            if per == 0:
                break
            result = result + chr(per)
        return result

    def connect_device(self, deviceList, index):
        stDeviceList = cast(deviceList.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            self.error_signal.emit("创建句柄失败! ret[0x%x]" % ret)
            return False

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.error_signal.emit("打开设备失败! ret[0x%x]" % ret)
            return False

        if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        self.connected_signal.emit(True)
        return True

    def set_trigger_mode(self, trigger_source="Line0"):
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0:
            self.error_signal.emit("设置触发模式失败! ret[0x%x]" % ret)
            return False

        source_map = {
            "Line0": MV_TRIGGER_SOURCE_LINE0,
            "Line1": MV_TRIGGER_SOURCE_LINE1,
            "Line2": MV_TRIGGER_SOURCE_LINE2,
            "Software": MV_TRIGGER_SOURCE_SOFTWARE,
        }
        
        source_value = source_map.get(trigger_source, MV_TRIGGER_SOURCE_LINE0)
        ret = self.cam.MV_CC_SetEnumValue("TriggerSource", source_value)
        if ret != 0:
            self.error_signal.emit("设置触发源失败! ret[0x%x]" % ret)
            return False

        return True

    def software_trigger_once(self):
        if self.cam:
            ret = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
            if ret != 0:
                self.error_signal.emit("软触发失败! ret[0x%x]" % ret)
                return False
            with self.trigger_lock:
                self.trigger_time = time.time()
            return True
        return False

    def start_grabbing(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.error_signal.emit("开始取流失败! ret[0x%x]" % ret)
            return False
        
        self.is_grabbing = True
        self.exit_flag = False
        self.grab_thread = threading.Thread(target=self._grab_thread_func)
        self.grab_thread.daemon = True
        self.grab_thread.start()
        return True

    def _grab_thread_func(self):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        
        while not self.exit_flag:
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret == 0 and stOutFrame.pBufAddr is not None:
                trigger_time = time.time()
                with self.trigger_lock:
                    if self.trigger_time:
                        trigger_time = self.trigger_time
                        self.trigger_time = None
                
                image_data = self._convert_to_opencv(stOutFrame)
                if image_data is not None:
                    with self.frame_lock:
                        self.latest_frame = image_data.copy()
                    self.frame_ready.emit(image_data, trigger_time)
                
                self.cam.MV_CC_FreeImageBuffer(stOutFrame)

    def _convert_to_opencv(self, stOutFrame):
        import cv2
        nWidth = stOutFrame.stFrameInfo.nWidth
        nHeight = stOutFrame.stFrameInfo.nHeight
        enPixelType = stOutFrame.stFrameInfo.enPixelType
        
        if self._is_mono(enPixelType):
            image_data = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
            cdll.msvcrt.memcpy(byref(image_data), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
            image = np.frombuffer(image_data, dtype=np.uint8)
            image = image.reshape((nHeight, nWidth))
            import cv2
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        elif self._is_color(enPixelType):
            nRGBSize = nWidth * nHeight * 3
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = nWidth
            stConvertParam.nHeight = nHeight
            stConvertParam.pSrcData = stOutFrame.pBufAddr
            stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = enPixelType
            stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
            stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
            stConvertParam.nDstBufferSize = nRGBSize
            
            ret = self.cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
            if ret != 0:
                return None
            
            image_data = (c_ubyte * stConvertParam.nDstLen)()
            cdll.msvcrt.memcpy(byref(image_data), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
            image = np.frombuffer(image_data, dtype=np.uint8)
            image = image.reshape((nHeight, nWidth, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return None

    def _is_mono(self, enPixelType):
        mono_types = [
            PixelType_Gvsp_Mono8, PixelType_Gvsp_Mono10, PixelType_Gvsp_Mono10_Packed,
            PixelType_Gvsp_Mono12, PixelType_Gvsp_Mono12_Packed
        ]
        return enPixelType in mono_types

    def _is_color(self, enPixelType):
        color_types = [
            PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerBG8, PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerRG10,
            PixelType_Gvsp_BayerGB10, PixelType_Gvsp_BayerBG10, PixelType_Gvsp_BayerGR12,
            PixelType_Gvsp_BayerRG12, PixelType_Gvsp_BayerGB12, PixelType_Gvsp_BayerBG12
        ]
        return enPixelType in color_types

    def stop_grabbing(self):
        self.exit_flag = True
        if self.is_grabbing:
            time.sleep(0.1)
            if self.cam:
                self.cam.MV_CC_StopGrabbing()
            self.is_grabbing = False

    def close_device(self):
        self.stop_grabbing()
        if self.cam:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
        self.connected_signal.emit(False)


class CameraCaptureDialog(QDialog):
    """相机采集对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.captured_image = None
        self.worker = CameraWorker()
        self.init_ui()
        self.init_camera()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("相机采集")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.image_label.setText("正在连接相机...")
        layout.addWidget(self.image_label)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.capture_btn = QPushButton("📷 截取图像")
        self.capture_btn.setMinimumHeight(40)
        self.capture_btn.setEnabled(False)  # 相机未就绪时禁用
        self.capture_btn.clicked.connect(self.capture_image)
        button_layout.addWidget(self.capture_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        # 状态标签
        self.status_label = QLabel("状态: 初始化中...")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def init_camera(self):
        """初始化相机连接"""
        # 连接信号
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.error_signal.connect(self.on_error)
        self.worker.connected_signal.connect(self.on_connected)

        # 搜索并连接设备
        result = self.worker.enum_devices()
        if result is None:
            QMessageBox.warning(self, "错误", "未找到相机设备!\n请检查相机连接和驱动。")
            self.reject()
            return

        self.devices, self.deviceList = result

        if len(self.devices) == 0:
            QMessageBox.warning(self, "错误", "未找到可用设备!")
            self.reject()
            return

        # 连接第一个设备
        if not self.worker.connect_device(self.deviceList, 0):
            QMessageBox.warning(self, "错误",
                                "连接相机失败!\n请检查:\n1. 相机是否被其他程序占用\n2. 防火墙设置\n3. IP地址配置")
            self.reject()
            return

        # 设置连续采集模式（不使用触发）
        self.set_continuous_mode()

        # 开始采集
        if not self.worker.start_grabbing():
            QMessageBox.warning(self, "错误", "开始采集失败!")
            self.worker.close_device()
            self.reject()
            return

        self.status_label.setText("状态: 相机就绪")
        self.capture_btn.setEnabled(True)

    def set_continuous_mode(self):
        """设置连续采集模式"""
        # 关闭触发模式，使用连续采集
        self.worker.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        # 设置采集帧率（可选）
        self.worker.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 30.0)

    def on_frame_ready(self, image, timestamp):
        """接收相机图像"""
        # 转换numpy数组为QPixmap显示
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # 缩放显示（保持比例）
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def on_error(self, msg):
        """错误处理"""
        self.status_label.setText(f"状态: 错误 - {msg}")

    def on_connected(self, status):
        """连接状态变化"""
        if not status:
            self.status_label.setText("状态: 相机已断开")

    def capture_image(self):
        """截取当前图像"""
        with self.worker.frame_lock:
            if self.worker.latest_frame is not None:
                self.captured_image = self.worker.latest_frame.copy()
                self.accept()  # 关闭对话框并返回Accept
            else:
                QMessageBox.warning(self, "提示", "暂无图像数据，请等待相机稳定后重试")

    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        if hasattr(self, 'worker'):
            self.worker.stop_grabbing()
            self.worker.close_device()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)

    # 测试CameraCaptureDialog
    dialog = CameraCaptureDialog()

    if dialog.exec_() == QDialog.Accepted and dialog.captured_image is not None:
        print(f"采集成功! 图像尺寸: {dialog.captured_image.shape}")

        # 保存测试图像
        import cv2

        cv2.imwrite("test_capture.jpg", dialog.captured_image)
        print("图像已保存为 test_capture.jpg")

        # 显示图像确认
        cv2.imshow("Captured Image", dialog.captured_image)
        print("按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        QMessageBox.information(None, "成功", "相机采集测试成功!")
    else:
        print("用户取消或采集失败")

    sys.exit(0)