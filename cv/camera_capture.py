"""
摄像头图像捕捉模块
"""
import cv2
import threading
import time
from datetime import datetime
import os


class CameraCapture:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.callbacks = []

    def start(self):
        """启动摄像头"""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def _capture_loop(self):
        """捕获循环"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame

                # 触发回调
                for callback in self.callbacks:
                    try:
                        callback(frame.copy())
                    except Exception as e:
                        print(f"Callback error: {e}")

            time.sleep(0.03)  # ~30fps

    def get_frame(self):
        """获取当前帧"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def capture_image(self, save_path=None):
        """捕获单张图片"""
        frame = self.get_frame()
        if frame is not None:
            if save_path:
                cv2.imwrite(save_path, frame)
            return frame
        return None

    def save_snapshot(self, prefix="snapshot"):
        """保存快照"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"

        # 确保目录存在
        os.makedirs("captures", exist_ok=True)
        filepath = os.path.join("captures", filename)

        self.capture_image(filepath)
        return filepath

    def register_callback(self, callback):
        """注册帧处理回调"""
        self.callbacks.append(callback)

    def stop(self):
        """停止摄像头"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()