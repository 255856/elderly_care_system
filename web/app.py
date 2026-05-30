"""
Flask 应用主文件
所有路由、蓝图注册和页面
"""
import os
import cv2
import numpy as np
import json
import base64
import shutil
import glob
from datetime import datetime

from flask import Flask, render_template, jsonify, send_from_directory, request, Response
from flask_login import LoginManager, login_required, current_user

from config import Config

login_manager = LoginManager()

# ── 提取人脸编码 ───────────────────────────────────────────

def extract_face_encoding_from_file(image_path):
    """从图片文件提取人脸特征，返回 (encoding, bbox)"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]
    face_roi = cv2.resize(face_roi, (128, 128))
    gx = cv2.Sobel(face_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(face_roi, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    hist = cv2.calcHist([ang.astype(np.uint8)], [0], None, [36], [0, 360])
    hist = hist.flatten()
    if np.linalg.norm(hist) > 0:
        hist = hist / np.linalg.norm(hist)
    return hist, faces[0]


def load_faces_from_db():
    """从数据库加载人脸特征到全局内存缓存"""
    global known_faces, face_id_counter
    from web.models import FaceRecord
    known_faces = {}
    face_id_counter = 1
    try:
        records = FaceRecord.query.all()
        for record in records:
            encoding = []
            if record.face_encoding:
                try:
                    encoding = json.loads(record.face_encoding)
                except Exception:
                    encoding = []
            known_faces[record.id] = {
                'encoding': encoding,
                'name': record.person_name,
                'type': record.person_type,
                'type_name': ('老年人' if record.person_type == 'elderly'
                              else ('工作人员' if record.person_type == 'staff' else '义工')),
                'person_id': record.person_id,
                'image_path': record.image_path,
                'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S') if record.created_at else ''
            }
            if record.id >= face_id_counter:
                face_id_counter = record.id + 1
        print(f"从数据库加载了 {len(records)} 个人脸记录")
    except Exception as e:
        print(f"加载人脸数据失败: {e}")


def get_gpu_info():
    """获取 GPU 信息"""
    import torch
    if torch.cuda.is_available():
        return {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        }
    return {'available': False, 'name': 'CPU', 'memory': 'N/A'}


# ── 应用工厂 ───────────────────────────────────────────────

known_faces = {}
face_id_counter = 1


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 确保目录存在
    for sub in ['static/faces', 'static/login_bgs', 'captures', 'backups', 'instance', 'uploads']:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    # 初始化数据库
    from web.models import db
    db.init_app(app)

    # 初始化登录管理
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = '请先登录'

    @login_manager.user_loader
    def load_user(user_id):
        from web.models import User
        return User.query.get(int(user_id))

    # 注册蓝图
    from web.routes.auth import auth_bp
    from web.routes.dashboard import dashboard_bp
    from web.routes.elderly import elderly_bp
    from web.routes.staff import staff_bp
    from web.routes.volunteer import volunteer_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(elderly_bp, url_prefix='/elderly')
    app.register_blueprint(staff_bp, url_prefix='/staff')
    app.register_blueprint(volunteer_bp, url_prefix='/volunteer')

    # ── 上下文处理器 ──
    @app.context_processor
    def inject_system_config():
        from web.models import SystemConfig
        configs = {}
        try:
            for c in SystemConfig.query.all():
                configs[c.config_key] = c.config_value
        except Exception as e:
            print(f"加载系统配置失败: {e}")
        return dict(system_config=configs)

    # ═══════════════════════════════════════════════════════
    # 系统配置 API
    # ═══════════════════════════════════════════════════════

    @app.route('/api/system-config', methods=['GET', 'POST'])
    def system_config_api():
        from web.models import SystemConfig, db
        if request.method == 'GET':
            configs = {}
            try:
                for c in SystemConfig.query.all():
                    configs[c.config_key] = c.config_value
                return jsonify({'success': True, 'configs': configs})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})

        data = request.get_json()
        if not data or 'key' not in data:
            return jsonify({'success': False, 'message': '缺少key参数'})
        key = data['key']
        value = data.get('value', '')

        if key != 'login_background' and not current_user.is_authenticated:
            return jsonify({'success': False, 'message': '请先登录'}), 401

        try:
            config = SystemConfig.query.filter_by(config_key=key).first()
            if config:
                config.config_value = value
                config.updated_at = datetime.now()
            else:
                config = SystemConfig(config_key=key, config_value=value)
                db.session.add(config)
            db.session.commit()
            return jsonify({'success': True, 'message': '保存成功'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': str(e)})

    @app.route('/api/upload-login-bg', methods=['POST'])
    def upload_login_bg():
        from web.models import SystemConfig, db
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': '没有上传文件'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名为空'})

        allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in allowed:
            return jsonify({'success': False, 'message': '只支持 png, jpg, jpeg, gif, webp'})

        file.seek(0, 2)
        if file.tell() > 5 * 1024 * 1024:
            return jsonify({'success': False, 'message': '图片大小不能超过5MB'})
        file.seek(0)

        bg_dir = os.path.join(app.root_path, 'static', 'login_bgs')
        os.makedirs(bg_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"login_bg_{timestamp}.{ext}"
        filepath = os.path.join(bg_dir, filename)
        file.save(filepath)
        bg_url = f'/static/login_bgs/{filename}'
        bg_value = f'url({bg_url})'

        try:
            config = SystemConfig.query.filter_by(config_key='login_background').first()
            if config:
                config.config_value = bg_value
                config.updated_at = datetime.now()
            else:
                config = SystemConfig(config_key='login_background', config_value=bg_value)
                db.session.add(config)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"保存配置失败: {e}")
        return jsonify({'success': True, 'path': bg_url, 'message': '上传成功'})

    @app.route('/api/detection-config', methods=['GET', 'POST'])
    @login_required
    def detection_config_api():
        from web.models import DetectionConfig, db
        if request.method == 'GET':
            try:
                config = DetectionConfig.query.first()
                if config:
                    return jsonify({'success': True, 'config': config.to_dict()})
                return jsonify({'success': True, 'config': {
                    'face_threshold': 0.6, 'stranger_threshold': 0.5, 'fall_sensitivity': 0.7
                }})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '没有数据'})
        try:
            config = DetectionConfig.query.first()
            if config:
                config.face_threshold = data.get('face_threshold', 0.6)
                config.stranger_threshold = data.get('stranger_threshold', 0.5)
                config.fall_sensitivity = data.get('fall_sensitivity', 0.7)
            else:
                config = DetectionConfig(
                    face_threshold=data.get('face_threshold', 0.6),
                    stranger_threshold=data.get('stranger_threshold', 0.5),
                    fall_sensitivity=data.get('fall_sensitivity', 0.7))
                db.session.add(config)
            db.session.commit()
            return jsonify({'success': True, 'message': '保存成功'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': str(e)})

    # ═══════════════════════════════════════════════════════
    # 页面路由
    # ═══════════════════════════════════════════════════════

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dashboard')
    @login_required
    def dashboard():
        return render_template('dashboard.html', user=current_user)

    @app.route('/elderly')
    @login_required
    def elderly_page():
        return render_template('elderly.html', user=current_user)

    @app.route('/staff')
    @login_required
    def staff_page():
        return render_template('staff.html', user=current_user)

    @app.route('/volunteer')
    @login_required
    def volunteer_page():
        return render_template('volunteer.html', user=current_user)

    @app.route('/monitor')
    @login_required
    def monitor():
        return render_template('monitor.html', user=current_user)

    @app.route('/report')
    @login_required
    def report():
        return render_template('report.html', user=current_user)

    @app.route('/alerts')
    @login_required
    def alerts_page():
        return render_template('alerts.html', user=current_user)

    @app.route('/settings')
    @login_required
    def settings():
        return render_template('settings.html', user=current_user)

    # ═══════════════════════════════════════════════════════
    # 摄像头 API - 使用 ElderlyCareSystem
    # ═══════════════════════════════════════════════════════

    def _get_system():
        from main import get_system
        return get_system(app=app)

    @app.route('/api/camera/start', methods=['POST'])
    @login_required
    def camera_start():
        try:
            success = _get_system().start_camera()
            if success:
                return jsonify({'success': True, 'message': '摄像头启动成功'})
            return jsonify({'success': False, 'message': '无法打开摄像头'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    @app.route('/api/camera/stop', methods=['POST'])
    @login_required
    def camera_stop():
        _get_system().stop_camera()
        return jsonify({'success': True, 'message': '摄像头已停止'})

    @app.route('/api/camera/status')
    @login_required
    def camera_status():
        system = _get_system()
        return jsonify({'is_opened': system.is_running})

    @app.route('/api/camera/capture', methods=['POST'])
    @login_required
    def camera_capture():
        system = _get_system()
        frame = system.get_frame()
        if frame is None:
            return jsonify({'success': False, 'message': '无法获取画面，请先启动摄像头'})
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'capture_{timestamp}.jpg'
        filepath = os.path.join(base_dir, 'captures', filename)
        cv2.imwrite(filepath, frame)
        return jsonify({'success': True, 'path': f'/captures/{filename}', 'message': '拍照成功'})

    @app.route('/api/camera/detect', methods=['GET'])
    @login_required
    def camera_detect():
        """执行真实 CV 检测"""

        system = _get_system()
        frame = system.get_frame()
        if frame is None:
            return jsonify({
                'success': True, 'face_count': 0, 'emotion': 'neutral',
                'confidence': 0.5, 'stranger': False, 'fall_detected': False
            })

        results = system.detect(frame)

        face_count = results['face_count']
        faces_list = results['faces']

        emotion = 'neutral'
        confidence = 0.5
        if results['emotions']:
            emo = results['emotions'][0]
            emotion = emo.get('emotion', 'neutral')
            confidence = emo.get('confidence', 0.5)
            if confidence > 0.65 and emotion in ('sad', 'angry', 'fear'):
                emotion_names = {'sad': '悲伤', 'angry': '生气', 'fear': '恐惧'}
                system.add_alert('情感分析', f'检测到{emotion_names.get(emotion, emotion)}情绪，建议关注老人状态',
                                 'info', cooldown=30)

        stranger_detected = len(results['strangers']) > 0
        if stranger_detected:
            system.add_alert('陌生人识别', f'检测到陌生人进入区域({face_count}人)，请注意观察',
                             'danger', cooldown=20)

        if face_count > 3:
            system.add_alert('人脸检测', f'检测到大量人群({face_count}人)，请注意观察',
                             'warning', cooldown=10)

        fall_detected = len(results['falls']) > 0
        if fall_detected:
            system.add_alert('摔倒检测', '检测到疑似摔倒行为，请立即查看监控画面！',
                             'danger', cooldown=60)

        return jsonify({
            'success': True,
            'face_count': face_count,
            'emotion': emotion,
            'confidence': min(confidence, 0.98),
            'stranger': stranger_detected,
            'fall_detected': fall_detected
        })

    @app.route('/api/camera/detect-frame', methods=['POST'])
    @login_required
    def camera_detect_frame():
        """从上传的图片帧执行检测"""
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'success': False, 'error': '没有图片数据'})

            image_data = data['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'success': False, 'error': '图片解码失败'})

            system = _get_system()
            annotated_frame, results = system.detect_with_overlay(frame)

            emotion = 'neutral'
            confidence = 0.5
            if results['emotions']:
                emo = results['emotions'][0]
                emotion = emo.get('emotion', 'neutral')
                confidence = emo.get('confidence', 0.5)

            return jsonify({
                'success': True,
                'face_count': results['face_count'],
                'emotion': emotion,
                'confidence': min(confidence, 0.98),
                'stranger': len(results['strangers']) > 0,
                'fall_detected': len(results['falls']) > 0
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/video_feed')
    @login_required
    def video_feed():
        system = _get_system()
        if not system.is_running:
            # 尝试自动启动摄像头
            system.start_camera()

        def generate():
            return system.generate_video_stream()

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # ═══════════════════════════════════════════════════════
    # GPU 信息
    # ═══════════════════════════════════════════════════════

    @app.route('/api/gpu-info')
    @login_required
    def gpu_info():
        return jsonify(get_gpu_info())

    # ═══════════════════════════════════════════════════════
    # 人脸管理 API
    # ═══════════════════════════════════════════════════════

    @app.route('/api/faces', methods=['GET'])
    @login_required
    def get_faces():
        from web.models import FaceRecord
        records = FaceRecord.query.order_by(FaceRecord.created_at.desc()).all()
        faces = [{
            'id': r.id, 'name': r.person_name, 'type': r.person_type,
            'type_name': ('老年人' if r.person_type == 'elderly'
                          else ('工作人员' if r.person_type == 'staff' else '义工')),
            'person_id': r.person_id,
            'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S') if r.created_at else ''
        } for r in records]
        return jsonify({'success': True, 'faces': faces})

    @app.route('/api/faces/add', methods=['POST'])
    @login_required
    def add_face():
        from web.models import FaceRecord, User, Elderly, Volunteer, db
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': '没有上传文件'})
        file = request.files['image']
        person_type = request.form.get('type')
        person_id = request.form.get('person_id')
        if not person_id:
            return jsonify({'success': False, 'message': '请选择人员'})
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名为空'})

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"face_{person_type}_{person_id}_{timestamp}.jpg"
        filepath = os.path.join(base_dir, 'static', 'faces', filename)
        file.save(filepath)

        encoding, _ = extract_face_encoding_from_file(filepath)
        if encoding is None:
            return jsonify({'success': False, 'message': '未检测到人脸，请上传清晰的人脸照片'})

        name = "未知"
        if person_type == 'elderly':
            person = Elderly.query.get(int(person_id))
            if person:
                name = person.name
        elif person_type == 'staff':
            person = User.query.get(int(person_id))
            if person:
                name = person.name
        elif person_type == 'volunteer':
            person = Volunteer.query.get(int(person_id))
            if person:
                name = person.name

        face_record = FaceRecord(
            person_type=person_type, person_id=int(person_id),
            person_name=name, face_encoding=json.dumps(encoding.tolist()),
            image_path=filepath, created_at=datetime.now())
        db.session.add(face_record)
        db.session.commit()

        # 重新加载到系统
        _get_system().reload_known_faces()
        load_faces_from_db()
        return jsonify({'success': True, 'message': '添加成功', 'face_id': face_record.id})

    @app.route('/api/faces/<int:face_id>', methods=['DELETE'])
    @login_required
    def delete_face(face_id):
        from web.models import FaceRecord, db
        face = FaceRecord.query.get(face_id)
        if not face:
            return jsonify({'success': False, 'message': '人脸不存在'})
        if face.image_path and os.path.exists(face.image_path):
            try:
                os.remove(face.image_path)
            except Exception:
                pass
        db.session.delete(face)
        db.session.commit()
        _get_system().reload_known_faces()
        load_faces_from_db()
        return jsonify({'success': True, 'message': '删除成功'})

    # ═══════════════════════════════════════════════════════
    # 备份 API
    # ═══════════════════════════════════════════════════════

    @app.route('/api/backups', methods=['GET'])
    @login_required
    def list_backups():
        backups = []
        backups_dir = os.path.join(base_dir, 'backups')
        os.makedirs(backups_dir, exist_ok=True)
        for filepath in glob.glob(os.path.join(backups_dir, 'backup_*.db')):
            filename = os.path.basename(filepath)
            stat = os.stat(filepath)
            backups.append({
                'name': filename,
                'size': f"{stat.st_size / 1024:.1f} KB",
                'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
        backups.sort(key=lambda x: x['date'], reverse=True)
        return jsonify({'success': True, 'backups': backups})

    @app.route('/api/backup', methods=['POST'])
    @login_required
    def backup_database():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'backup_{timestamp}.db'
        backups_dir = os.path.join(base_dir, 'backups')
        os.makedirs(backups_dir, exist_ok=True)
        backup_path = os.path.join(backups_dir, backup_filename)

        # 从 SQLAlchemy 引擎获取实际的数据库文件路径
        from web.models import db as _db
        db_url = str(_db.engine.url)
        if db_url.startswith('sqlite:///'):
            db_path = db_url[len('sqlite:///'):]
            if db_path.startswith('/') and len(db_path) > 2 and db_path[2] == ':':
                db_path = db_path[1:]
        else:
            db_path = os.path.join(app.instance_path, 'elderly_care.db')
        db_path = os.path.normpath(db_path)

        try:
            if os.path.exists(db_path):
                shutil.copy2(db_path, backup_path)
                return jsonify({'success': True, 'filename': backup_filename,
                                'path': f'/backups/{backup_filename}', 'message': '备份成功'})
            return jsonify({'success': False, 'message': '数据库文件不存在'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    @app.route('/api/restore', methods=['POST'])
    @login_required
    def restore_database():
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有上传文件'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名为空'})
        if not file.filename.endswith('.db'):
            return jsonify({'success': False, 'message': '请选择.db备份文件'})

        # 简单校验：检查 SQLite 文件头
        header = file.read(16)
        file.seek(0)
        if header[:16] != b'SQLite format 3\x00':
            return jsonify({'success': False, 'message': '无效的数据库文件'})

        backups_dir = os.path.join(base_dir, 'backups')
        os.makedirs(backups_dir, exist_ok=True)
        temp_path = os.path.join(backups_dir, 'temp_restore.db')
        file.save(temp_path)

        # 从 SQLAlchemy 引擎获取实际的数据库文件路径
        from web.models import db as _db
        db_url = str(_db.engine.url)
        if db_url.startswith('sqlite:///'):
            db_path = db_url[len('sqlite:///'):]
            # Windows 绝对路径兼容: sqlite:///E:/path 或 sqlite:////E:/path
            if db_path.startswith('/') and len(db_path) > 2 and db_path[2] == ':':
                db_path = db_path[1:]  # 去掉前导 /
        else:
            db_path = os.path.join(app.instance_path, 'elderly_care.db')
        db_path = os.path.normpath(db_path)

        try:
            # 先关闭当前数据库连接，否则覆盖文件后连接仍指向旧库
            _db.session.remove()
            _db.engine.dispose()

            # 覆盖数据库文件
            shutil.copy2(temp_path, db_path)
            os.remove(temp_path)

            # 重新建立连接并加载数据
            with app.app_context():
                _db.create_all()
            load_faces_from_db()
            _get_system().reload_known_faces()

            return jsonify({'success': True, 'message': '数据恢复成功'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    @app.route('/backups/<filename>')
    @login_required
    def download_backup(filename):
        backups_dir = os.path.join(base_dir, 'backups')
        return send_from_directory(backups_dir, filename)

    @app.route('/captures/<filename>')
    def capture_file(filename):
        captures_dir = os.path.join(base_dir, 'captures')
        return send_from_directory(captures_dir, filename)

    # ═══════════════════════════════════════════════════════
    # 其他 API
    # ═══════════════════════════════════════════════════════

    @app.route('/api/stats')
    @login_required
    def api_stats():
        from web.models import Elderly, User, Volunteer
        return jsonify({
            'elderly_count': Elderly.query.filter_by(status='active').count(),
            'staff_count': User.query.filter_by(status='active').count(),
            'volunteer_count': Volunteer.query.filter_by(status='active').count(),
            'alert_count': 0
        })

    @app.route('/health')
    def health():
        import torch
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'gpu_available': torch.cuda.is_available()
        })

    @app.route('/test')
    def test():
        return jsonify({'status': 'ok', 'message': '服务器连接正常'})

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        uploads_dir = os.path.join(base_dir, 'uploads')
        return send_from_directory(uploads_dir, filename)

    @app.errorhandler(404)
    def not_found(error):
        if request.is_json:
            return jsonify({'error': '页面不存在', 'code': 404}), 404
        return "<h1>404 - 页面未找到</h1><p>您访问的页面不存在</p><a href='/'>返回首页</a>", 404

    @app.errorhandler(500)
    def internal_error(error):
        if request.is_json:
            return jsonify({'error': '服务器内部错误', 'code': 500}), 500
        return "<h1>500 - 服务器错误</h1><p>服务器出了点问题</p><a href='/'>返回首页</a>", 500

    # ── 应用启动初始化 ──
    with app.app_context():
        from web.models import db as _db, User
        _db.create_all()

        # 确保管理员账户存在
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin', name='系统管理员',
                role='admin', department='管理部', status='active'
            )
            admin.set_password('admin123')
            _db.session.add(admin)
            _db.session.commit()
            print("默认管理员已创建 (admin / admin123)")

        load_faces_from_db()
        from main import init_system
        init_system(app)
        print("系统初始化完成")

    return app


app = create_app()
