"""
老年人管理路由
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from web.models import db, Elderly
from datetime import datetime

elderly_bp = Blueprint('elderly', __name__)


@elderly_bp.route('/')
@login_required
def index():
    """老年人管理页面"""
    return render_template('elderly.html', user=current_user)


@elderly_bp.route('/list')
@login_required
def get_list():
    """获取老年人列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    keyword = request.args.get('keyword', '')

    query = Elderly.query.filter_by(status='active')

    if keyword:
        query = query.filter(
            db.or_(
                Elderly.name.like(f'%{keyword}%'),
                Elderly.id_card.like(f'%{keyword}%'),
                Elderly.phone.like(f'%{keyword}%')
            )
        )

    pagination = query.order_by(Elderly.id.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'items': [e.to_dict() for e in pagination.items],
        'total': pagination.total,
        'page': page,
        'pages': pagination.pages,
        'per_page': per_page
    })


@elderly_bp.route('/create', methods=['POST'])
@login_required
def create():
    """添加老年人"""
    data = request.get_json()

    if not data.get('name', '').strip():
        return jsonify({'success': False, 'message': '姓名不能为空'}), 400

    try:
        # 空字符串转为 None，避免 unique 约束冲突
        id_card = data.get('id_card', '').strip() or None

        elderly = Elderly(
            name=data.get('name', '').strip(),
            gender=data.get('gender'),
            age=data.get('age') if data.get('age') else None,
            id_card=id_card,
            phone=data.get('phone', '').strip() or None,
            emergency_contact=data.get('emergency_contact', '').strip() or None,
            emergency_phone=data.get('emergency_phone', '').strip() or None,
            address=data.get('address', '').strip() or None,
            health_status=data.get('health_status', '').strip() or None,
            room_number=data.get('room_number', '').strip() or None,
            admission_date=datetime.strptime(data.get('admission_date'), '%Y-%m-%d') if data.get(
                'admission_date') else None
        )

        db.session.add(elderly)
        db.session.commit()

        return jsonify({'success': True, 'message': '添加成功', 'id': elderly.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@elderly_bp.route('/update/<int:id>', methods=['PUT'])
@login_required
def update(id):
    """更新老年人信息"""
    elderly = Elderly.query.get_or_404(id)
    data = request.get_json()

    try:
        for key, value in data.items():
            if hasattr(elderly, key) and value is not None:
                if key == 'admission_date' and value:
                    value = datetime.strptime(value, '%Y-%m-%d')
                if key == 'id_card' and isinstance(value, str) and value.strip() == '':
                    value = None
                setattr(elderly, key, value)

        db.session.commit()
        return jsonify({'success': True, 'message': '更新成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@elderly_bp.route('/delete/<int:id>', methods=['DELETE'])
@login_required
def delete(id):
    """删除老年人（软删除）"""
    elderly = Elderly.query.get_or_404(id)
    elderly.status = 'deleted'
    db.session.commit()

    return jsonify({'success': True, 'message': '删除成功'})


@elderly_bp.route('/<int:id>')
@login_required
def get_detail(id):
    """获取老年人详情"""
    elderly = Elderly.query.get_or_404(id)
    return jsonify(elderly.to_dict())


@elderly_bp.route('/statistics')
@login_required
def get_statistics():
    """获取统计数据"""
    total = Elderly.query.filter_by(status='active').count()

    male_count = Elderly.query.filter_by(status='active', gender='男').count()
    female_count = Elderly.query.filter_by(status='active', gender='女').count()

    # 年龄分布
    age_groups = {
        '60-70': Elderly.query.filter(Elderly.age >= 60, Elderly.age < 70, Elderly.status == 'active').count(),
        '70-80': Elderly.query.filter(Elderly.age >= 70, Elderly.age < 80, Elderly.status == 'active').count(),
        '80-90': Elderly.query.filter(Elderly.age >= 80, Elderly.age < 90, Elderly.status == 'active').count(),
        '90以上': Elderly.query.filter(Elderly.age >= 90, Elderly.status == 'active').count()
    }

    # 房间分布
    rooms = db.session.query(Elderly.room_number, db.func.count(Elderly.id)) \
        .filter(Elderly.status == 'active', Elderly.room_number.isnot(None)) \
        .group_by(Elderly.room_number).all()

    return jsonify({
        'total': total,
        'gender': {'male': male_count, 'female': female_count},
        'age_groups': age_groups,
        'rooms': [{'room': r[0], 'count': r[1]} for r in rooms]
    })