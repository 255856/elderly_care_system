"""
数据模型 - 修复版
"""
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# 创建数据库实例（稍后在app中初始化）
db = SQLAlchemy()

class User(UserMixin, db.Model):
    """用户模型（工作人员）"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    role = db.Column(db.String(50), default='staff')
    department = db.Column(db.String(100))
    avatar_path = db.Column(db.String(200))
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def set_password(self, password):
        """设置密码哈希"""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password, password)

    def get_id(self):
        """Flask-Login需要的方法"""
        return str(self.id)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'name': self.name,
            'role': self.role,
            'department': self.department,
            'status': self.status
        }


class Elderly(db.Model):
    """老年人模型"""
    __tablename__ = 'elderly'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    id_card = db.Column(db.String(18), unique=True)
    phone = db.Column(db.String(20))
    emergency_contact = db.Column(db.String(100))
    emergency_phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    health_status = db.Column(db.Text)
    room_number = db.Column(db.String(20))
    avatar_path = db.Column(db.String(200))
    admission_date = db.Column(db.Date)
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'gender': self.gender,
            'age': self.age,
            'id_card': self.id_card,
            'phone': self.phone,
            'emergency_contact': self.emergency_contact,
            'emergency_phone': self.emergency_phone,
            'address': self.address,
            'health_status': self.health_status,
            'room_number': self.room_number,
            'admission_date': self.admission_date.strftime('%Y-%m-%d') if self.admission_date else '',
            'status': self.status
        }


class Volunteer(db.Model):
    """义工模型"""
    __tablename__ = 'volunteer'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    id_card = db.Column(db.String(18))
    skills = db.Column(db.Text)
    available_time = db.Column(db.Text)
    avatar_path = db.Column(db.String(200))
    register_date = db.Column(db.Date)
    total_hours = db.Column(db.Float, default=0)
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'gender': self.gender,
            'age': self.age,
            'phone': self.phone,
            'email': self.email,
            'id_card': self.id_card,
            'skills': self.skills,
            'available_time': self.available_time,
            'total_hours': self.total_hours,
            'status': self.status
        }

class Alert(db.Model):
    """告警记录模型"""
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50))  # 告警类型：摔倒检测、陌生人识别、情感分析等
    severity = db.Column(db.String(20), default='warning')  # 严重程度：danger, warning, info
    description = db.Column(db.Text)  # 告警描述
    is_resolved = db.Column(db.Boolean, default=False)  # 是否已处理
    created_at = db.Column(db.DateTime, default=datetime.now)  # 创建时间
    resolved_at = db.Column(db.DateTime, nullable=True)  # 处理时间

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'severity': self.severity,
            'description': self.description,
            'is_resolved': self.is_resolved,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else '',
            'resolved_at': self.resolved_at.strftime('%Y-%m-%d %H:%M:%S') if self.resolved_at else ''
        }


class FaceRecord(db.Model):
    """人脸记录模型"""
    __tablename__ = 'face_records'

    id = db.Column(db.Integer, primary_key=True)
    person_type = db.Column(db.String(20))  # elderly, staff, volunteer
    person_id = db.Column(db.Integer)
    person_name = db.Column(db.String(100))
    face_encoding = db.Column(db.Text)  # 存储人脸特征向量
    image_path = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'person_type': self.person_type,
            'person_id': self.person_id,
            'person_name': self.person_name,
            'image_path': self.image_path,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }

class SystemConfig(db.Model):
    __tablename__ = 'system_configs'
    id = db.Column(db.Integer, primary_key=True)
    config_key = db.Column(db.String(100), unique=True, nullable=False)
    config_value = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)


class DetectionConfig(db.Model):
    """检测参数配置模型"""
    __tablename__ = 'detection_configs'

    id = db.Column(db.Integer, primary_key=True)
    face_threshold = db.Column(db.Float, default=0.6)
    stranger_threshold = db.Column(db.Float, default=0.5)
    fall_sensitivity = db.Column(db.Float, default=0.7)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            'face_threshold': self.face_threshold,
            'stranger_threshold': self.stranger_threshold,
            'fall_sensitivity': self.fall_sensitivity
        }