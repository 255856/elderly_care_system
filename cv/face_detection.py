"""
人脸检测模块 - 优化版
支持：YuNet (默认)、Haar Cascade、MediaPipe、MTCNN、dlib、YOLOv8
"""
import cv2
import numpy as np
from collections import deque

from cv.yolo_face_detector import YunetFaceDetector

yunet_detector = None
box_smoothing_cache = {}
smoothing_window = 5


def get_yunet_detector():
    """获取 YuNet 检测器单例"""
    global yunet_detector
    if yunet_detector is None:
        try:
            yunet_detector = YunetFaceDetector()
            print("YuNet 人脸检测器初始化成功")
        except Exception as e:
            print(f"YuNet 检测器初始化失败: {e}")
            yunet_detector = None
    return yunet_detector


def smooth_boxes(boxes, person_id=0):
    """平滑检测框，减少闪烁"""
    global box_smoothing_cache
    if person_id not in box_smoothing_cache:
        box_smoothing_cache[person_id] = deque(maxlen=smoothing_window)
    if len(boxes) == 0:
        return []
    box = boxes[0]
    box_smoothing_cache[person_id].append(box)
    if len(box_smoothing_cache[person_id]) > 0:
        boxes_list = list(box_smoothing_cache[person_id])
        x1 = int(np.mean([b[0] for b in boxes_list]))
        y1 = int(np.mean([b[1] for b in boxes_list]))
        x2 = int(np.mean([b[2] for b in boxes_list]))
        y2 = int(np.mean([b[3] for b in boxes_list]))
        return [(x1, y1, x2, y2)]
    return boxes


def clear_smoothing_cache():
    global box_smoothing_cache
    box_smoothing_cache.clear()


def cleanup_stale_smoothing_entries(max_entries=20):
    """清理长期未使用的平滑缓存条目，防止内存泄漏"""
    global box_smoothing_cache
    if len(box_smoothing_cache) > max_entries:
        excess = len(box_smoothing_cache) - max_entries
        keys_to_remove = list(box_smoothing_cache.keys())[:excess]
        for key in keys_to_remove:
            del box_smoothing_cache[key]


class FaceDetector:
    def __init__(self, method='yunet'):
        """
        初始化人脸检测器
        method: 'yunet' (默认), 'haar', 'mediapipe', 'mtcnn', 'dlib', 'yolov8'
        """
        self.method = method
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if self.method == 'yunet':
            self.detector = get_yunet_detector()
            if self.detector is None:
                print("YuNet 不可用，回退到 Haar")
                self.method = 'haar'
                self._init_detector()

        elif self.method == 'haar':
            haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(haar_path)

        elif self.method == 'dlib':
            import dlib
            self.detector = dlib.get_frontal_face_detector()

        elif self.method == 'mtcnn':
            from mtcnn import MTCNN
            self.detector = MTCNN()

        elif self.method == 'mediapipe':
            self._init_mediapipe()

        elif self.method == 'yolov8':
            from cv.yolo_face_detector import PersonDetector
            self.detector = PersonDetector()
            if self.detector.model is None:
                print("YOLOv8 不可用，回退到 YuNet")
                self.method = 'yunet'
                self._init_detector()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            face_detection = mp.solutions.face_detection
            self.detector = face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
            print("MediaPipe 人脸检测加载成功")
        except Exception as e:
            print(f"MediaPipe 加载失败: {e}，回退到 Haar")
            self.method = 'haar'
            self._init_detector()

    def detect_yunet(self, image, use_smoothing=True):
        """YuNet 人脸检测"""
        if self.detector is None:
            return []
        faces = self.detector.detect(image)
        if use_smoothing and len(faces) > 0:
            faces = smooth_boxes(faces)
        return faces

    def detect_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        return [(x, y, x + w, y + h) for (x, y, w, h) in faces]

    def detect_dlib(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return [(rect.left(), rect.top(), rect.right(), rect.bottom()) for rect in faces]

    def detect_mtcnn(self, image):
        results = self.detector.detect_faces(image)
        faces = []
        for result in results:
            x, y, w, h = result['box']
            if w > 0 and h > 0 and w > 50 and h > 50:
                faces.append((x, y, x + w, y + h))
        return faces

    def detect_mediapipe(self, image):
        if self.detector is None:
            return []
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)
            faces = []
            if results.detections:
                h, w = image.shape[:2]
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    fw = int(bbox.width * w)
                    fh = int(bbox.height * h)
                    if fw > 50 and fh > 50:
                        x, y = max(0, x), max(0, y)
                        faces.append((x, y, x + fw, y + fh))
            return faces
        except Exception as e:
            print(f"MediaPipe 检测失败: {e}")
            return []

    def detect_yolov8(self, image, use_smoothing=True):
        """YOLOv8 person 检测（返回检测到的人体框，不推荐用于人脸）"""
        if self.detector is None:
            return []
        try:
            faces = self.detector.detect(image, conf_threshold=0.5)
            if use_smoothing and len(faces) > 0:
                faces = smooth_boxes(faces)
            return faces
        except Exception as e:
            print(f"YOLOv8 检测失败: {e}")
            return []

    def detect(self, image, use_smoothing=True):
        """统一检测接口"""
        if self.method == 'yunet':
            return self.detect_yunet(image, use_smoothing)
        elif self.method == 'haar':
            return self.detect_haar(image)
        elif self.method == 'dlib':
            return self.detect_dlib(image)
        elif self.method == 'mtcnn':
            return self.detect_mtcnn(image)
        elif self.method == 'mediapipe':
            return self.detect_mediapipe(image)
        elif self.method == 'yolov8':
            return self.detect_yolov8(image, use_smoothing)
        return []

    def get_face_locations(self, image):
        return self.detect(image)


class FaceRecognition:
    """人脸识别类"""
    def __init__(self):
        self.face_recognition = None
        self.known_faces = {}
        self._init_recognizer()

    def _init_recognizer(self):
        try:
            import face_recognition
            self.face_recognition = face_recognition
            print("face_recognition 加载成功")
        except ImportError:
            print("face_recognition not installed, using fallback method")
            self.face_recognition = None

    def load_known_faces(self, face_records):
        """从外部加载已知人脸编码"""
        import json
        self.known_faces.clear()
        for record in face_records:
            encoding = record.get('encoding')
            if isinstance(encoding, str):
                try:
                    encoding = json.loads(encoding)
                except Exception:
                    encoding = None
            if encoding is not None:
                self.known_faces[record.get('id')] = {
                    'encoding': encoding,
                    'name': record.get('name', ''),
                    'type': record.get('type', '')
                }

    def extract_hog_features(self, face_image):
        """提取 HOG 特征作为回退方案"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        hist = cv2.calcHist([ang.astype(np.uint8)], [0], None, [36], [0, 360])
        hist = hist.flatten()
        if np.linalg.norm(hist) > 0:
            hist = hist / np.linalg.norm(hist)
        return hist

    def get_face_encoding(self, image, face_location):
        """获取人脸编码"""
        x1, y1, x2, y2 = face_location
        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
        if self.face_recognition:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            encodings = self.face_recognition.face_encodings(rgb)
            if encodings:
                return encodings[0]
        return self.extract_hog_features(face_img)

    def compare_faces(self, known_encoding, face_encoding, tolerance=0.6):
        """比较两个人脸是否匹配"""
        if known_encoding is None or face_encoding is None:
            return False, 0
        if self.face_recognition:
            result = self.face_recognition.compare_faces([known_encoding], face_encoding, tolerance)
            return result[0] if result else False, 1.0

        known = np.array(known_encoding)
        face = np.array(face_encoding)
        min_len = min(len(known), len(face))
        known, face = known[:min_len], face[:min_len]
        dot = np.dot(known, face)
        norm1, norm2 = np.linalg.norm(known), np.linalg.norm(face)
        if norm1 == 0 or norm2 == 0:
            return False, 0
        similarity = dot / (norm1 * norm2)
        return similarity >= tolerance, similarity
