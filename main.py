"""
智慧养老系统 - 主入口
唯一的 CV 模块管理和摄像头控制入口，web 层通过 get_system() 获取实例
"""
import threading
import time
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv.camera_capture import CameraCapture
from cv.face_detection import FaceDetector, FaceRecognition, cleanup_stale_smoothing_entries
from cv.emotion_analysis import EmotionAnalyzer
from cv.fall_detection import FallDetector
from cv.stranger_recognition import StrangerRecognizer
from cv.intrusion_detection import IntrusionDetector
from cv.yolo_face_detector import PersonDetector, YOLOPoseDetector


class ElderlyCareSystem:
    """智慧养老系统主类 - CV模块和摄像头的唯一管理入口"""

    def __init__(self, app=None):
        self.app = app

        self.yolo_face = None
        self.yolo_pose = None
        self.face_detector = None
        self.face_recognizer = None
        self.emotion_analyzer = None
        self.fall_detector = None
        self.stranger_recognizer = None
        self.intrusion_detector = None

        self.camera = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False

        self._last_alert_time = {}
        self._init_modules()

    # ── 初始化 ─────────────────────────────────────────────

    def _init_modules(self):
        print("=" * 50)
        print("初始化智慧养老系统 CV 模块...")

        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}  "
                  f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("CPU 模式")

        try:
            self.yolo_face = PersonDetector(device='auto')
            self.face_detector = FaceDetector(method='yunet')
            print("✅ 人脸检测器 (YuNet + YOLOv8)")
        except Exception as e:
            print(f"⚠️ YOLOv8 人脸检测失败: {e}，回退 Haar")
            self.face_detector = FaceDetector(method='haar')

        try:
            self.yolo_pose = YOLOPoseDetector(device='auto')
            self.fall_detector = FallDetector(method='yolov8')
            print("✅ 摔倒检测器 (YOLOv8 Pose)")
        except Exception as e:
            print(f"⚠️ YOLOv8 Pose 失败: {e}，回退简单模式")
            self.fall_detector = FallDetector(method='simple')

        try:
            self.face_recognizer = FaceRecognition()
            print("✅ 人脸识别器")
        except Exception as e:
            print(f"⚠️ 人脸识别失败: {e}")

        try:
            self.emotion_analyzer = EmotionAnalyzer()
            print("✅ 情感分析器")
        except Exception as e:
            print(f"⚠️ 情感分析失败: {e}")

        self.stranger_recognizer = StrangerRecognizer()
        self._reload_known_faces()
        print("✅ 陌生人识别器")

        self.intrusion_detector = IntrusionDetector()
        print("✅ 入侵检测器")
        print("=" * 50)

    # ── 已知人脸管理 ───────────────────────────────────────

    def _reload_known_faces(self):
        """从数据库重新加载已知人脸"""
        if self.app is None:
            return
        try:
            with self.app.app_context():
                from web.models import FaceRecord
                records = FaceRecord.query.all()
                encodings = []
                info_list = []
                import json
                for r in records:
                    if r.face_encoding:
                        try:
                            enc = json.loads(r.face_encoding)
                        except Exception:
                            continue
                        encodings.append(enc)
                        info_list.append({
                            'type': r.person_type,
                            'id': r.person_id,
                            'name': r.person_name
                        })
                self.stranger_recognizer.load_faces(encodings, info_list)
                print(f"已加载 {len(encodings)} 个已知人脸")
        except Exception as e:
            print(f"加载已知人脸失败: {e}")

    def reload_known_faces(self):
        """公开接口：重新加载已知人脸"""
        self._reload_known_faces()

    # ── 摄像头控制 ─────────────────────────────────────────

    def start_camera(self, camera_index=0):
        if self.is_running:
            return True
        self.camera = CameraCapture(camera_index=camera_index)
        self.camera.register_callback(self._on_frame)
        self.camera.start()
        self.is_running = True
        return True

    def stop_camera(self):
        self.is_running = False
        if self.camera:
            self.camera.stop()
            self.camera = None

    def _on_frame(self, frame):
        """摄像头帧回调：保存最新帧并执行检测"""
        with self.frame_lock:
            self.current_frame = frame.copy()

        cleanup_stale_smoothing_entries()

    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    # ── 检测接口 ───────────────────────────────────────────

    def detect(self, frame):
        """对单帧执行完整检测，返回结构化结果"""
        results = {
            'faces': [],
            'face_count': 0,
            'emotions': [],
            'strangers': [],
            'falls': [],
            'intrusions': [],
        }
        if frame is None:
            return results

        faces = self.face_detector.detect(frame) if self.face_detector else []
        results['face_count'] = len(faces)

        for (x1, y1, x2, y2) in faces:
            face_info = {'bbox': (x1, y1, x2, y2)}
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            if self.face_recognizer:
                encoding = self.face_recognizer.get_face_encoding(frame, (x1, y1, x2, y2))
                if encoding is not None and self.stranger_recognizer:
                    is_known, person_info, sim = self.stranger_recognizer.recognize(encoding)
                    face_info['is_known'] = is_known
                    face_info['person_info'] = person_info
                    face_info['similarity'] = sim
                    if not is_known:
                        results['strangers'].append(face_info)

            if self.emotion_analyzer:
                emo = self.emotion_analyzer.analyze_emotion(face_img)
                if emo:
                    face_info['emotion'] = emo
                    results['emotions'].append(emo)

            results['faces'].append(face_info)

        # 摔倒检测（YOLOv8 Pose）
        if self.fall_detector and self.yolo_pose:
            try:
                poses = self.yolo_pose.detect_poses(frame, conf_threshold=0.5)
                for kp in poses:
                    is_fall, conf = self.yolo_pose.is_falling(kp, frame.shape)
                    if is_fall:
                        results['falls'].append({'confidence': conf})
            except Exception as e:
                print(f"摔倒检测出错: {e}")

        # 入侵检测
        if self.intrusion_detector and self.intrusion_detector.forbidden_zones:
            person_bboxes = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in faces]
            intrusions = self.intrusion_detector.detect_intrusion(frame, person_bboxes)
            results['intrusions'] = intrusions

        return results

    def detect_with_overlay(self, frame):
        """检测并在图像上绘制标注框，返回 (带标注的图像, 检测结果)"""
        results = self.detect(frame)

        for face in results['faces']:
            x1, y1, x2, y2 = face['bbox']
            color = (0, 0, 255) if not face.get('is_known', True) else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_parts = []
            if face.get('person_info') and face['person_info'].get('name'):
                label_parts.append(face['person_info']['name'])
            elif not face.get('is_known', True):
                label_parts.append("陌生人")

            if face.get('emotion'):
                label_parts.append(f"{face['emotion']['emotion']}:{face['emotion']['confidence']:.2f}")

            if label_parts:
                cv2.putText(frame, ' '.join(label_parts), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.intrusion_detector:
            frame = self.intrusion_detector.draw_forbidden_zones(frame)

        return frame, results

    def generate_video_stream(self):
        """MJPEG 视频流生成器"""
        from datetime import datetime

        frame_count = 0
        detection_interval = 5

        while self.is_running:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1

            if frame_count % detection_interval == 0:
                results = self.detect(frame)
            else:
                results = {'faces': [], 'face_count': 0, 'falls': []}

            for face in results['faces']:
                x1, y1, x2, y2 = face['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"Faces: {results['face_count']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if results['falls']:
                cv2.putText(frame, 'FALL DETECTED!', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def capture_snapshot(self):
        if self.camera:
            return self.camera.save_snapshot()
        return None

    # ── 告警 ───────────────────────────────────────────────

    def add_alert(self, alert_type, description, severity='warning', cooldown=30):
        """添加告警（带限频），写入 SQLAlchemy 数据库"""
        current = time.time()
        key = f"{alert_type}:{severity}"
        last = self._last_alert_time.get(key, 0)
        if current - last < cooldown:
            return None

        self._last_alert_time[key] = current

        if self.app:
            try:
                with self.app.app_context():
                    from web.models import db, Alert
                    from datetime import datetime
                    alert = Alert(
                        type=alert_type, severity=severity,
                        description=description, is_resolved=False,
                        created_at=datetime.now()
                    )
                    db.session.add(alert)
                    db.session.commit()
                    return alert
            except Exception as e:
                print(f"添加告警失败: {e}")
        return None


# ── 全局单例 ───────────────────────────────────────────────

_system = None
_system_lock = threading.Lock()


def get_system(app=None):
    """获取系统单例（线程安全）"""
    global _system
    if _system is None:
        with _system_lock:
            if _system is None:
                _system = ElderlyCareSystem(app=app)
    return _system


def init_system(app):
    """在 Flask 应用上下文中初始化系统"""
    return get_system(app=app)
