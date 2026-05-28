"""
摔倒监测模块 - 优化版
使用 YOLOv8 Pose 进行人体姿态估计
"""
import cv2
import numpy as np
import os

# 导入 YOLOv8 Pose 检测器
from cv.yolo_face_detector import YOLOPoseDetector

pose_detector = None


def get_pose_detector():
    """获取 YOLOv8 Pose 检测器单例"""
    global pose_detector
    if pose_detector is None:
        try:
            pose_detector = YOLOPoseDetector()
            print("YOLOv8 Pose 检测器初始化成功")
        except Exception as e:
            print(f"YOLOv8 Pose 检测器初始化失败: {e}")
            pose_detector = None
    return pose_detector


class FallDetector:
    def __init__(self, method='yolov8'):
        """
        初始化摔倒检测器
        method: 'yolov8', 'mediapipe', 'simple'
        """
        self.method = method
        self.aspect_ratio_threshold = 0.5
        self.fall_history = []
        self.history_size = 5
        self.yolo_pose = None
        self._init_detector()

    def _init_detector(self):
        if self.method == 'yolov8':
            self.yolo_pose = get_pose_detector()
            if self.yolo_pose is None:
                print("YOLOv8 Pose 不可用，回退到简单模式")
                self.method = 'simple'
        elif self.method == 'mediapipe':
            self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp = mp
            pose = mp.solutions.pose
            self.pose = pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Pose 加载成功")
        except Exception as e:
            print(f"MediaPipe Pose 加载失败: {e}")
            self.method = 'simple'

    def detect_fall(self, frame, person_bbox=None):
        """
        检测是否发生摔倒

        Returns:
            (is_fall: bool, confidence: float, details: dict)
        """
        try:
            if self.method == 'yolov8' and self.yolo_pose:
                return self._detect_fall_yolo(frame)
            elif hasattr(self, 'pose') and self.pose:
                return self._detect_fall_mediapipe(frame)
            else:
                return self._analyze_aspect_ratio_fall(person_bbox)
        except Exception as e:
            print(f"Fall detection error: {e}")
            return False, 0.0, {'error': str(e)}

    def _detect_fall_yolo(self, frame):
        """使用 YOLOv8 Pose 检测摔倒"""
        try:
            poses = self.yolo_pose.detect_poses(frame, conf_threshold=0.5)

            if len(poses) == 0:
                return False, 0.0, {'error': 'no pose detected'}

            for keypoints in poses:
                is_fall, confidence = self.yolo_pose.is_falling(keypoints, frame.shape)
                if is_fall:
                    return True, confidence, {'reason': 'yolo_pose_fall_detected'}

            return False, 0.0, {}
        except Exception as e:
            print(f"YOLOv8 Pose 检测失败: {e}")
            return False, 0.0, {}

    def _detect_fall_mediapipe(self, frame):
        """使用 MediaPipe 姿态估计检测摔倒"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if not results.pose_landmarks:
                return False, 0.0, {}

            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # 获取关键点
            left_shoulder = (landmarks[11].x * w, landmarks[11].y * h)
            right_shoulder = (landmarks[12].x * w, landmarks[12].y * h)
            left_hip = (landmarks[23].x * w, landmarks[23].y * h)
            right_hip = (landmarks[24].x * w, landmarks[24].y * h)

            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                              (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2,
                         (left_hip[1] + right_hip[1]) / 2)

            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            angle = np.arctan2(abs(dy), abs(dx)) * 180 / np.pi if dx != 0 else 90

            if angle < 30:
                return True, min(0.9, 1 - angle / 90), {'body_angle': angle}

            return False, 0.0, {'body_angle': angle}
        except Exception as e:
            print(f"MediaPipe 检测失败: {e}")
            return False, 0.0, {}

    def _analyze_aspect_ratio_fall(self, bbox):
        """基于宽高比分析摔倒"""
        if bbox is None:
            return False, 0.0, {}

        x, y, w, h = bbox
        if h == 0:
            return False, 0.0, {}

        aspect_ratio = w / h

        if aspect_ratio > self.aspect_ratio_threshold * 2:
            confidence = min(0.8, aspect_ratio / 3)
            return True, confidence, {'aspect_ratio': aspect_ratio}

        return False, 0.0, {'aspect_ratio': aspect_ratio}

    def analyze_motion_velocity(self, prev_bbox, curr_bbox):
        """分析运动速度"""
        if prev_bbox is None or curr_bbox is None:
            return 0

        prev_center = (prev_bbox[0] + prev_bbox[2] // 2,
                       prev_bbox[1] + prev_bbox[3] // 2)
        curr_center = (curr_bbox[0] + curr_bbox[2] // 2,
                       curr_bbox[1] + curr_bbox[3] // 2)

        velocity = np.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                          (curr_center[1] - prev_center[1]) ** 2)

        self.fall_history.append(velocity)
        if len(self.fall_history) > self.history_size:
            self.fall_history.pop(0)

        return velocity