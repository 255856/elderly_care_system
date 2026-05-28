"""
系统配置文件
"""
import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'elderly-care-system-dev-key'

    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///elderly_care.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 文件上传配置
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # 人脸图片存储路径
    FACE_IMAGES_PATH = 'static/faces/'

    # 摄像头配置
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # 检测阈值
    FALL_DETECTION_THRESHOLD = 0.7
    INTRUSION_CONFIDENCE = 0.6
    STRANGER_THRESHOLD = 0.5
