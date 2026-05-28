"""
陌生人识别模块
"""
import numpy as np


class StrangerRecognizer:
    def __init__(self, threshold=0.6):
        self.known_faces = []
        self.known_person_info = []
        self.threshold = threshold

    def load_faces(self, encodings, person_info_list):
        """加载已知人脸编码和对应的人员信息"""
        self.known_faces = [np.array(e) for e in encodings]
        self.known_person_info = person_info_list

    def recognize(self, face_encoding):
        """
        识别人脸

        Returns:
            (is_known: bool, person_info: dict, similarity: float)
        """
        if not self.known_faces or face_encoding is None:
            return False, None, 0.0

        similarities = []
        for known_encoding in self.known_faces:
            sim = self._cosine_similarity(face_encoding, known_encoding)
            similarities.append(sim)

        if not similarities:
            return False, None, 0.0

        max_sim = max(similarities)
        max_idx = similarities.index(max_sim)

        if max_sim > self.threshold:
            return True, self.known_person_info[max_idx], max_sim

        return False, None, max_sim

    def _cosine_similarity(self, encoding1, encoding2):
        """计算余弦相似度"""
        e1 = np.array(encoding1)
        e2 = np.array(encoding2)

        dot = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))
