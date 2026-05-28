"""
工作人员管理路由 - 修复版
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from web.models import db, User

staff_bp = Blueprint('staff', __name__)


@staff_bp.route('/')
@login_required
def index():
    """工作人员管理页面"""
    return render_template('staff.html', user=current_user)


@staff_bp.route('/list')
@login_required
def get_list():
    """获取工作人员列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    keyword = request.args.get('keyword', '')

    query = User.query.filter(User.status == 'active')

    if keyword:
        query = query.filter(
            db.or_(
                User.name.like(f'%{keyword}%'),
                User.username.like(f'%{keyword}%'),
                User.phone.like(f'%{keyword}%')
            )
        )

    pagination = query.order_by(User.id.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    # 转换为字典，避免敏感信息
    items = []
    for u in pagination.items:
        items.append({
            'id': u.id,
            'username': u.username,
            'name': u.name,
            'gender': u.gender,
            'age': u.age,
            'phone': u.phone,
            'email': u.email,
            'role': u.role,
            'department': u.department,
            'status': u.status
        })

    return jsonify({
        'items': items,
        'total': pagination.total,
        'page': page,
        'pages': pagination.pages,
        'per_page': per_page
    })


@staff_bp.route('/create', methods=['POST'])
@login_required
def create():
    """添加工作人员"""
    data = request.get_json()

    # 检查用户名是否已存在
    if User.query.filter_by(username=data.get('username')).first():
        return jsonify({'success': False, 'message': '用户名已存在'}), 400

    try:
        staff = User(
            username=data.get('username'),
            name=data.get('name'),
            gender=data.get('gender'),
            age=data.get('age'),
            phone=data.get('phone'),
            email=data.get('email'),
            role=data.get('role', 'staff'),
            department=data.get('department'),
            status='active'
        )
        staff.set_password(data.get('password', '123456'))

        db.session.add(staff)
        db.session.commit()

        return jsonify({'success': True, 'message': '添加成功', 'id': staff.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@staff_bp.route('/update/<int:id>', methods=['PUT'])
@login_required
def update(id):
    """更新工作人员信息"""
    staff = User.query.get_or_404(id)
    data = request.get_json()

    try:
        # 允许更新的字段
        allowed_fields = ['name', 'gender', 'age', 'phone', 'email', 'role', 'department']
        for key, value in data.items():
            if key in allowed_fields and value is not None:
                setattr(staff, key, value)

        # 如果提供了新密码
        if data.get('password'):
            staff.set_password(data.get('password'))

        db.session.commit()
        return jsonify({'success': True, 'message': '更新成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400


@staff_bp.route('/delete/<int:id>', methods=['DELETE'])
@login_required
def delete(id):
    """删除工作人员"""
    if id == current_user.id:
        return jsonify({'success': False, 'message': '不能删除自己的账户'}), 400

    staff = User.query.get_or_404(id)
    staff.status = 'deleted'
    db.session.commit()

    return jsonify({'success': True, 'message': '删除成功'})


@staff_bp.route('/<int:id>')
@login_required
def get_detail(id):
    """获取工作人员详情"""
    staff = User.query.get_or_404(id)
    return jsonify({
        'id': staff.id,
        'username': staff.username,
        'name': staff.name,
        'gender': staff.gender,
        'age': staff.age,
        'phone': staff.phone,
        'email': staff.email,
        'role': staff.role,
        'department': staff.department
    })


@staff_bp.route('/statistics')
@login_required
def get_statistics():
    """获取统计信息"""
    total = User.query.filter_by(status='active').count()

    role_stats = db.session.query(User.role, db.func.count(User.id)) \
        .filter(User.status == 'active') \
        .group_by(User.role).all()

    dept_stats = db.session.query(User.department, db.func.count(User.id)) \
        .filter(User.status == 'active', User.department.isnot(None)) \
        .group_by(User.department).all()

    return jsonify({
        'total': total,
        'by_role': [{'role': r[0] or 'staff', 'count': r[1]} for r in role_stats],
        'by_department': [{'department': d[0], 'count': d[1]} for d in dept_stats]
    })