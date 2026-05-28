"""
禁止区域入侵检测 - YOLOv8 人体检测版
"""
import cv2
import numpy as np


class IntrusionDetector:
    def __init__(self):
        self.forbidden_zones = []

    def add_forbidden_zone(self, points, name=None):
        """
        添加禁止区域

        Args:
            points: 多边形顶点列表 [(x1,y1), (x2,y2), ...]
            name: 区域名称
        """
        self.forbidden_zones.append({
            'points': np.array(points, np.int32),
            'name': name or f"Zone_{len(self.forbidden_zones)}"
        })

    def set_forbidden_zones_from_config(self, zones):
        """从配置设置禁止区域"""
        for zone in zones:
            self.add_forbidden_zone(zone['points'], zone.get('name'))

    def detect_intrusion(self, frame, persons):
        """
        检测入侵

        Args:
            frame: 当前帧（未使用，保留以兼容）
            persons: bbox 列表 [(x, y, w, h), ...] 或 dict 列表 [{'bbox': ...}, ...]

        Returns:
            List of intrusion events
        """
        intrusions = []

        for person in persons:
            if isinstance(person, dict):
                bbox = person['bbox']
                person_center = person.get('center', self._get_bbox_center(bbox))
                confidence = person.get('confidence', 1.0)
            else:
                bbox = person
                person_center = self._get_bbox_center(bbox)
                confidence = 1.0

            for zone in self.forbidden_zones:
                if self._point_in_polygon(person_center, zone['points']):
                    intrusions.append({
                        'zone_name': zone['name'],
                        'person_location': person_center,
                        'bbox': bbox,
                        'confidence': confidence,
                        'timestamp': cv2.getTickCount()
                    })

        return intrusions

    def _get_bbox_center(self, bbox):
        """从 bbox 计算中心"""
        x1, y1, x2, y2 = bbox
        return (int((x1+x2)/2), int((y1+y2)/2))

    def _point_in_polygon(self, point, polygon):
        """射线法判断点是否在多边形内"""
        x, y = point
        inside = False
        n = len(polygon)

        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def draw_forbidden_zones(self, frame):
        """在图像上绘制禁止区域"""
        for zone in self.forbidden_zones:
            cv2.polylines(frame, [zone['points']], True, (0, 0, 255), 2)

            # 半透明填充
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone['points']], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # 添加区域标签
            center = np.mean(zone['points'], axis=0).astype(int)
            cv2.putText(frame, zone['name'], tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame