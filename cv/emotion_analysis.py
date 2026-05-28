"""
情感分析模块 - 表情识别
"""
import cv2
import numpy as np


class EmotionAnalyzer:
    def __init__(self):
        self.emotions = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral','disgust']
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化情感分析模型"""
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
        except ImportError:
            print("DeepFace not installed, using fallback")
            self.deepface = None

    def analyze_emotion(self, face_image):
        """
        分析单张人脸的情感

        Returns:
            dict: {'emotion': 'happy', 'confidence': 0.95, 'all_scores': {...}}
        """
        if self.deepface:
            try:
                result = self.deepface.analyze(face_image, actions=['emotion'], enforce_detection=False)
                if result:
                    emotions = result[0]['emotion']
                    dominant = result[0]['dominant_emotion']
                    return {
                        'emotion': dominant,
                        'confidence': emotions[dominant] / 100,
                        'all_scores': emotions
                    }
            except Exception as e:
                print(f"Emotion analysis error: {e}")

        # 模拟响应（实际使用时应替换为真实模型）
        return self._mock_analysis(face_image)

    def _mock_analysis(self, face_image):
        """模拟情感分析（用于测试）"""
        import random
        emotion = random.choice(self.emotions)
        return {
            'emotion': emotion,
            'confidence': random.uniform(0.6, 0.95),
            'all_scores': {e: random.uniform(0, 0.5) for e in self.emotions}
        }

    def analyze_interaction(self, elderly_face, volunteer_face):
        """
        分析老人和义工的互动情感

        Returns:
            dict: 互动分析结果
        """
        elderly_emotion = self.analyze_emotion(elderly_face)
        volunteer_emotion = self.analyze_emotion(volunteer_face)

        # 判断互动质量
        positive_emotions = ['happy', 'neutral', 'surprise']
        is_positive = (elderly_emotion['emotion'] in positive_emotions and
                       volunteer_emotion['emotion'] in positive_emotions)

        return {
            'elderly': elderly_emotion,
            'volunteer': volunteer_emotion,
            'interaction_quality': 'positive' if is_positive else 'needs_attention',
            'suggestion': '互动良好' if is_positive else '建议关注老人情绪'
        }