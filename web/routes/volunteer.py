"""
义工管理路由 - 修复版
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from web.models import db, Volunteer
from datetime import datetime

volunteer_bp = Blueprint('volunteer', __name__)


@volunteer_bp.route('/')
@login_required
def index():
    """义工管理页面"""
    return render_template('volunteer.html', user=current_user)


@volunteer_bp.route('/list')
@login_required
def get_list():
    """获取义工列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    keyword = request.args.get('keyword', '')

    query = Volunteer.query.filter_by(status='active')

    if keyword:
        query = query.filter(
            db.or_(
                Volunteer.name.like(f'%{keyword}%'),
                Volunteer.phone.like(f'%{keyword}%'),
                Volunteer.skills.like(f'%{keyword}%')
            )
        )

    pagination = query.order_by(Volunteer.total_hours.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'items': [v.to_dict() for v in pagination.items],
        'total': pagination.total,
        'page': page,
        'pages': pagination.pages,
        'per_page': per_page
    })


@volunteer_bp.route('/create', methods=['POST'])
@login_required
def create():
    """添加义工"""
    data = request.get_json()

    try:
        volunteer = Volunteer(
            name=data.get('name'),
            gender=data.get('gender'),
            age=data.get('age'),
            phone=data.get('phone'),
            email=data.get('email'),
            id_card=data.get('id_card'),
            skills=data.get('skills'),
            available_time=data.get('available_time'),
            register_date=datetime.now().date(),
            total_hours=0,
            status='active'
        )

        db.session.add(volunteer)
        db.session.commit()

        return jsonify({'success': True, 'message': '添加成功', 'id': volunteer.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@volunteer_bp.route('/update/<int:id>', methods=['PUT'])
@login_required
def update(id):
    """更新义工信息"""
    volunteer = Volunteer.query.get_or_404(id)
    data = request.get_json()

    try:
        # 只更新存在的字段
        allowed_fields = ['name', 'gender', 'age', 'phone', 'email', 'id_card', 'skills', 'available_time']
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(volunteer, key, value)

        db.session.commit()
        return jsonify({'success': True, 'message': '更新成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@volunteer_bp.route('/delete/<int:id>', methods=['DELETE'])
@login_required
def delete(id):
    """删除义工"""
    volunteer = Volunteer.query.get_or_404(id)
    volunteer.status = 'deleted'
    db.session.commit()

    return jsonify({'success': True, 'message': '删除成功'})


@volunteer_bp.route('/<int:id>')
@login_required
def get_detail(id):
    """获取义工详情"""
    volunteer = Volunteer.query.get_or_404(id)
    return jsonify(volunteer.to_dict())


@volunteer_bp.route('/add-hours/<int:id>', methods=['POST'])
@login_required
def add_hours(id):
    """增加服务时长"""
    volunteer = Volunteer.query.get_or_404(id)
    data = request.get_json()
    hours = data.get('hours', 0)

    if hours <= 0:
        return jsonify({'success': False, 'message': '时长必须大于0'}), 400

    volunteer.total_hours = (volunteer.total_hours or 0) + hours
    db.session.commit()

    return jsonify({'success': True, 'message': f'增加 {hours} 小时', 'total_hours': volunteer.total_hours})


@volunteer_bp.route('/statistics')
@login_required
def get_statistics():
    """获取统计信息"""
    total = Volunteer.query.filter_by(status='active').count()

    # 性别分布
    male_count = Volunteer.query.filter_by(status='active', gender='男').count()
    female_count = Volunteer.query.filter_by(status='active', gender='女').count()

    # 总服务时长
    total_hours = db.session.query(db.func.sum(Volunteer.total_hours)).filter_by(status='active').scalar() or 0

    return jsonify({
        'total': total,
        'gender': {'male': male_count, 'female': female_count},
        'total_hours': float(total_hours),
        'avg_hours': float(total_hours / total) if total > 0 else 0
    })