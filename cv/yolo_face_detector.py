"""
人体检测与姿态估计模块
- PersonDetector: YOLOv8 人体检测（用于摔倒检测、入侵检测）
- YOLOPoseDetector: YOLOv8 姿态估计（用于摔倒判定）
- YunetFaceDetector: OpenCV YuNet 人脸检测（高精度、轻量级）
"""
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from collections import deque


# ═══════════════════════════════════════════════════════════
# YOLOv8 人体检测器（用于摔倒、入侵等场景）
# ═══════════════════════════════════════════════════════════

class PersonDetector:
    """YOLOv8 人体检测器 - 检测图像中的 person (COCO class 0)"""

    def __init__(self, model_path='yolov8n.pt', device='auto'):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.model = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.frame_cache = None
        self.cache_counter = 0
        self.cache_max = 3
        self._init_model()

    def _resolve_device(self, device):
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _init_model(self):
        try:
            self.model = YOLO(self.model_path)
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            else:
                print("YOLOv8 人体检测使用 CPU")
        except Exception as e:
            print(f"YOLOv8 模型加载失败: {e}")
            self.model = None

    def detect(self, image, conf_threshold=0.5, force_detect=False):
        """检测人体，返回 [(x1, y1, x2, y2), ...]"""
        if self.model is None:
            return []

        if not force_detect:
            self.cache_counter += 1
            if self.cache_counter < self.cache_max:
                if self.frame_cache is not None:
                    return self.frame_cache
                return []
            self.cache_counter = 0

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, conf=conf_threshold, iou=self.iou_threshold, verbose=False)

        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else -1
                    if cls == 0:  # person
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        h, w = image.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1 and (x2 - x1) > 30 and (y2 - y1) > 30:
                            persons.append((x1, y1, x2, y2))

        self.frame_cache = persons
        return persons


# ═══════════════════════════════════════════════════════════
# YOLOv8 姿态检测器（用于摔倒判定）
# ═══════════════════════════════════════════════════════════

class YOLOPoseDetector:
    """YOLOv8 姿态检测器 - 17 关键点 COCO 格式"""

    def __init__(self, model_path='yolov8n-pose.pt', device='auto'):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.model = None
        self.pose_cache = None
        self.cache_counter = 0
        self.cache_max = 5
        self._init_model()

    def _resolve_device(self, device):
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _init_model(self):
        try:
            self.model = YOLO(self.model_path)
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            else:
                print("YOLOv8 Pose 使用 CPU")
        except Exception as e:
            print(f"YOLOv8 Pose 模型加载失败: {e}")
            self.model = None

    def detect_poses(self, image, conf_threshold=0.5, force_detect=False):
        """检测人体姿态，返回关键点列表"""
        if self.model is None:
            return []

        if not force_detect:
            self.cache_counter += 1
            if self.cache_counter < self.cache_max:
                if self.pose_cache is not None:
                    return self.pose_cache
                return []
            self.cache_counter = 0

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, conf=conf_threshold, verbose=False)

        poses = []
        for result in results:
            if result.keypoints is not None and result.keypoints.xy is not None:
                if len(result.keypoints.xy) > 0:
                    keypoints = result.keypoints.xy[0].cpu().numpy()
                    if len(keypoints) > 0:
                        poses.append(keypoints)


        self.pose_cache = poses
        return poses

    def is_falling(self, keypoints, image_shape, fall_threshold=0.4):
        """判断是否摔倒（基于肩-髋角度）"""
        if keypoints is None or len(keypoints) < 17:
            return False, 0.0

        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
        LEFT_HIP, RIGHT_HIP = 11, 12

        ls = keypoints[LEFT_SHOULDER]
        rs = keypoints[RIGHT_SHOULDER]
        lh = keypoints[LEFT_HIP]
        rh = keypoints[RIGHT_HIP]

        if ls[0] < 0 or rs[0] < 0 or lh[0] < 0 or rh[0] < 0:
            return False, 0.0

        shoulder_center = (ls + rs) / 2
        hip_center = (lh + rh) / 2

        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]

        angle = 90 if abs(dx) < 1e-6 else np.arctan2(abs(dy), abs(dx)) * 180 / np.pi

        is_fall = angle < 30
        confidence = min(0.9, 1 - angle / 90) if is_fall else 0.0
        return is_fall, confidence


# ═══════════════════════════════════════════════════════════
# OpenCV YuNet 人脸检测器（轻量、高精度、真正的脸检测）
# ═══════════════════════════════════════════════════════════

class YunetFaceDetector:
    """OpenCV YuNet 人脸检测器 - 基于 ONNX，真正的脸部检测"""

    def __init__(self,
                 model_path='face_detection_yunet_2023mar.onnx',
                 input_size=(320, 320),
                 conf_threshold=0.6,
                 nms_threshold=0.3,
                 top_k=5000):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            self.model = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config='',
                input_size=self.input_size,
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k
            )
            print(f"YuNet 人脸检测器加载成功 ({self.input_size[0]}x{self.input_size[1]})")
        except Exception as e:
            print(f"YuNet 加载失败: {e}")
            self.model = None

    def detect(self, image):
        """检测人脸，返回 [(x1, y1, x2, y2), ...]"""
        if self.model is None:
            return []

        h, w = image.shape[:2]
        self.model.setInputSize((w, h))

        try:
            _, faces = self.model.detect(image)
        except Exception as e:
            print(f"YuNet 检测失败: {e}")
            return []

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            # YuNet 返回: [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
            #              nose_x, nose_y, mouth_right_x, mouth_right_y,
            #              mouth_left_x, mouth_left_y, confidence]
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[14])

            # 坐标裁剪
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)

            if x2 > x and y2 > y and (x2 - x) > 20 and (y2 - y) > 20:
                results.append((x, y, x2, y2))

        return results


# ═══════════════════════════════════════════════════════════
# 向后兼容的别名
# ═══════════════════════════════════════════════════════════

YOLOFaceDetector = PersonDetector  # 向后兼容旧代码
