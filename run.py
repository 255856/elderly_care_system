"""
系统启动脚本
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.app import app

if __name__ == '__main__':
    print("启动 Web 服务器: http://localhost:5000")
    try:
        from waitress import serve
        print("使用 Waitress 生产服务器")
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        print("Waitress 未安装，使用 Flask 开发服务器")
        app.run(debug=False, host='0.0.0.0', port=5000)
