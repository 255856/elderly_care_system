"""
仪表板路由 - 完整版
"""
from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from web.models import db, Alert
from datetime import datetime

dashboard_bp = Blueprint('dashboard', __name__)


def add_alert(alert_type, description, severity='warning'):
    """添加告警到数据库"""
    try:
        alert = Alert(
            type=alert_type,
            severity=severity,
            description=description,
            is_resolved=False,
            created_at=datetime.now()
        )
        db.session.add(alert)
        db.session.commit()
        return alert
    except Exception as e:
        print(f"添加告警失败: {e}")
        db.session.rollback()
        return None


@dashboard_bp.route('/')
@login_required
def index():
    return render_template('dashboard.html', user=current_user)


@dashboard_bp.route('/stats')
@login_required
def get_stats():
    """获取统计数据"""
    from web.models import Elderly, User, Volunteer

    elderly_count = Elderly.query.filter_by(status='active').count()
    staff_count = User.query.filter_by(status='active').count()
    volunteer_count = Volunteer.query.filter_by(status='active').count()
    unresolved_count = Alert.query.filter_by(is_resolved=False).count()

    return jsonify({
        'elderly_count': elderly_count,
        'staff_count': staff_count,
        'volunteer_count': volunteer_count,
        'alert_count': unresolved_count
    })


@dashboard_bp.route('/alerts')
@login_required
def get_alerts():
    """获取告警列表"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    alert_type = request.args.get('type', '')
    status = request.args.get('status', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')

    # 构建查询
    query = Alert.query.order_by(Alert.created_at.desc())

    if alert_type:
        query = query.filter(Alert.type.contains(alert_type))

    if status == 'unresolved':
        query = query.filter(Alert.is_resolved == False)
    elif status == 'resolved':
        query = query.filter(Alert.is_resolved == True)

    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(Alert.created_at >= start)
        except:
            pass

    if end_date:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            query = query.filter(Alert.created_at <= end.replace(hour=23, minute=59, second=59))
        except:
            pass

    # 分页
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'items': [a.to_dict() for a in pagination.items],
        'total': pagination.total,
        'page': page,
        'pages': pagination.pages
    })


@dashboard_bp.route('/resolve-alert/<int:alert_id>', methods=['POST'])
@login_required
def resolve_alert(alert_id):
    """标记告警为已处理"""
    alert = Alert.query.get(alert_id)
    if not alert:
        return jsonify({'success': False, 'message': '告警不存在'})

    alert.is_resolved = True
    alert.resolved_at = datetime.now()
    db.session.commit()

    return jsonify({'success': True, 'message': '已标记为已处理'})


@dashboard_bp.route('/delete-alert/<int:alert_id>', methods=['DELETE'])
@login_required
def delete_alert(alert_id):
    """删除告警"""
    alert = Alert.query.get(alert_id)
    if not alert:
        return jsonify({'success': False, 'message': '告警不存在'})

    db.session.delete(alert)
    db.session.commit()

    return jsonify({'success': True, 'message': '删除成功'})


@dashboard_bp.route('/clear-alerts', methods=['POST'])
@login_required
def clear_alerts():
    """清空所有告警"""
    try:
        Alert.query.delete()
        db.session.commit()
        return jsonify({'success': True, 'message': '已清空所有告警'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})


@dashboard_bp.route('/detection_status')
@login_required
def detection_status():
    """获取检测状态（用于监控页面）"""
    # 获取最近5条未处理的告警
    recent_alerts = Alert.query.filter_by(is_resolved=False).order_by(Alert.created_at.desc()).limit(5).all()

    # 统计陌生人告警数量
    stranger_count = Alert.query.filter(
        Alert.type.like('%陌生人%'),
        Alert.is_resolved == False
    ).count()

    return jsonify({
        'results': [
            {'type': '人脸检测', 'result': '监控中', 'confidence': 0},
            {'type': '情感分析', 'result': '监控中', 'confidence': 0},
            {'type': '陌生人识别', 'result': f'{stranger_count} 次告警', 'confidence': 0},
            {'type': '摔倒检测', 'result': '监控中', 'confidence': 0}
        ],
        'alerts': [a.to_dict() for a in recent_alerts]
    })


# 导出添加告警的函数供其他模块使用
def add_system_alert(alert_type, description, severity='warning'):
    return add_alert(alert_type, description, severity)


# 在 dashboard.py 末尾添加以下代码

@dashboard_bp.route('/save-navbar-color', methods=['POST'])
@login_required
def save_navbar_color():
    """保存导航栏颜色"""
    from web.models import SystemConfig, db
    data = request.get_json()
    color = data.get('color', '')
    if not color:
        return jsonify({'success': False, 'message': '颜色值不能为空'})
    try:
        config = SystemConfig.query.filter_by(config_key='navbar_color').first()
        if config:
            config.config_value = color
        else:
            config = SystemConfig(config_key='navbar_color', config_value=color)
            db.session.add(config)
        db.session.commit()
        return jsonify({'success': True, 'message': '保存成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})


@dashboard_bp.route('/get-system-config', methods=['GET'])
@login_required
def get_system_config():
    """获取系统配置"""
    from web.models import SystemConfig
    configs = {}
    try:
        for c in SystemConfig.query.all():
            configs[c.config_key] = c.config_value
        return jsonify({'success': True, 'configs': configs})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
