"""
数据库初始化脚本 - 完整修复版
直接在PyCharm中运行此文件即可
"""
import os
import sys

# 添加项目路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 60)
print("智慧养老系统 - 数据库初始化")
print("=" * 60)


def init_database():
    """初始化数据库"""

    # 导入应用
    from web.app import app
    from web.models import db, User

    # 确保instance目录存在
    os.makedirs('instance', exist_ok=True)

    with app.app_context():
        print("\n📁 正在创建数据库表...")

        # 删除所有现有表（谨慎使用，会清空所有数据）
        # 取消下面两行的注释可以重置数据库
        # db.drop_all()
        # print("⚠️ 已删除所有现有表")

        # 创建所有表
        db.create_all()
        print("✅ 数据库表创建成功！")

        # 检查并创建管理员账户
        print("\n👤 检查管理员账户...")
        admin = User.query.filter_by(username='admin').first()

        if not admin:
            admin = User(
                username='admin',
                name='系统管理员',
                role='admin',
                department='管理部',
                status='active'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("✅ 管理员账户创建成功！")
            print("   用户名: admin")
            print("   密码: admin123")
        else:
            print("⚠️ 管理员账户已存在")
            print(f"   用户名: {admin.username}")
            print(f"   姓名: {admin.name}")

        # 创建示例工作人员（可选）
        print("\n👥 检查示例数据...")
        if User.query.count() == 1:  # 只有管理员
            demo_users = [
                ('zhangsan', '张三', 'staff', '护理部', '123456'),
                ('lisi', '李四', 'doctor', '医疗部', '123456'),
                ('wangwu', '王五', 'nurse', '护理部', '123456'),
            ]

            for username, name, role, dept, pwd in demo_users:
                if not User.query.filter_by(username=username).first():
                    user = User(
                        username=username,
                        name=name,
                        role=role,
                        department=dept,
                        status='active'
                    )
                    user.set_password(pwd)
                    db.session.add(user)

            db.session.commit()
            print("✅ 添加了 3 个示例工作人员")
        else:
            print("⚠️ 示例数据已存在或无需添加")

        # 显示所有表
        print("\n📋 数据库表列表:")
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        for i, table in enumerate(tables, 1):
            # 获取每个表的记录数
            try:
                count = db.session.execute(
                    db.text(f"SELECT COUNT(*) FROM \"{table}\"")
                ).scalar()
                print(f"   {i}. {table} ({count} 条记录)")
            except:
                print(f"   {i}. {table}")

        # 显示数据库文件位置
        db_path = os.path.join(current_dir, 'instance', 'elderly_care.db')
        print(f"\n💾 数据库文件位置:")
        print(f"   {db_path}")

        print("\n" + "=" * 60)
        print("✅ 数据库初始化完成！")
        print("=" * 60)
        print("\n下一步:")
        print("   1. 运行 run.py 启动系统")
        print("   2. 访问 http://localhost:5000")
        print("   3. 使用 admin / admin123 登录")
        print("=" * 60)


if __name__ == '__main__':
    try:
        init_database()
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback

        traceback.print_exc()

        print("\n💡 可能的解决方法:")
        print("   1. 确保已安装依赖: pip install flask flask-sqlalchemy flask-login")
        print("   2. 检查文件结构是否正确")
        print("   3. 确保在正确的目录下运行")

        input("\n按 Enter 键退出...")