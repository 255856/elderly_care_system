# 智慧养老系统 / Smart Elderly Care System

基于 Flask 和 OpenCV 的智慧养老管理系统，集成计算机视觉技术，提供老年人管理、工作人员管理、义工管理、智能监控、人脸识别、情感分析等功能。

A smart elderly care management system built with Flask and OpenCV, featuring real-time video monitoring, face recognition, emotion analysis, stranger detection, and fall detection.

## 功能模块

### 1. 仪表板
- 数据概览统计
- 实时告警展示
- 快捷入口导航

### 2. 老年人管理
- 老年人信息增删改查
- 健康信息管理
- 房间分配管理
- 入住记录管理

### 3. 工作人员管理
- 员工信息管理
- 角色权限分配（admin / staff / manager / doctor / nurse）
- 密码重置功能
- 部门管理

### 4. 义工管理
- 义工信息管理
- 服务时长统计
- 技能标签管理
- 服务记录追踪

### 5. 智能监控
- 实时视频监控
- 人脸检测（支持 Haar Cascade / YuNet / YOLOv8）
- 人脸识别（HOG 特征 + 余弦相似度）
- 情感分析（7 种表情识别）
- 陌生人识别
- 摔倒检测（YOLOv8 Pose）
- 入侵检测（多边形禁区）
- 实时拍照记录

### 6. 报表统计
- 数据可视化图表
- 人口统计分析
- 服务时长排行

### 7. 告警记录
- 实时告警同步
- 按类型 / 状态筛选
- 日期范围查询
- 告警处理与删除
- 自动刷新（3 / 5 / 10 / 30 秒可选）

### 8. 系统设置
- 个人信息管理
- 密码修改
- 人脸库管理
- 检测参数配置
- 数据备份与恢复

## 技术栈

### 后端
- **框架**: Flask 2.3+
- **ORM**: SQLAlchemy + Flask-SQLAlchemy
- **认证**: Flask-Login
- **计算机视觉**: OpenCV, YOLOv8, YuNet
- **深度学习**: PyTorch / Ultralytics
- **生产服务器**: Waitress

### 前端
- HTML5 + CSS3 + Vanilla JavaScript
- Fetch API 异步请求
- Chart.js 数据可视化

## 项目结构

```
elderly_care_system/
├── main.py                     # CV 系统核心类（单例模式）
├── run.py                      # 生产环境入口
├── init_db.py                  # 数据库初始化脚本
├── download_models.py          # 模型下载脚本
├── config.py                   # Flask 配置文件
├── requirements.txt            # Python 依赖
├── cv/                         # 计算机视觉模块
│   ├── camera_capture.py       # 摄像头管理（线程安全）
│   ├── face_detection.py       # 人脸检测与识别
│   ├── emotion_analysis.py     # 情感分析
│   ├── fall_detection.py       # 摔倒检测
│   ├── stranger_recognition.py # 陌生人识别
│   ├── intrusion_detection.py  # 入侵检测
│   └── yolo_face_detector.py   # YOLOv8 / YuNet 封装
├── web/                        # Flask Web 应用
│   ├── app.py                  # 应用工厂 + REST API
│   ├── models.py               # 数据库模型
│   ├── routes/                 # 蓝图路由
│   │   ├── auth.py             # 认证
│   │   ├── dashboard.py        # 仪表板
│   │   ├── elderly.py          # 老年人管理
│   │   ├── staff.py            # 工作人员管理
│   │   └── volunteer.py        # 义工管理
│   ├── static/
│   │   ├── faces/              # 注册人脸照片
│   │   └── login_bgs/          # 登录页背景图
│   └── templates/              # HTML 模板
├── instance/                   # SQLite 数据库（运行时生成）
├── captures/                   # 拍照保存（运行时生成）
├── backups/                    # 数据库备份（运行时生成）
└── uploads/                    # 文件上传（运行时生成）
```

## 安装与运行

### 环境要求
- Python 3.8+
- 摄像头设备（可选，用于智能监控）

### 安装步骤

**1. 克隆项目**
```bash
git clone https://github.com/255856/elderly_care_system.git
cd elderly_care_system
```

**2. 创建虚拟环境（推荐）**
```bash
# 使用 conda
conda create -n elderly python=3.9
conda activate elderly

# 或使用 venv
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

**3. 安装依赖**
```bash
pip install -r requirements.txt
```

**4. 下载模型文件**

YOLOv8 模型会在首次使用时自动下载。YuNet 人脸检测模型需手动下载：
```bash
python download_models.py
```

**5. 初始化数据库**
```bash
python init_db.py
```

**6. 启动系统**
```bash
python run.py
```

**7. 访问系统**
```
http://localhost:5000
```

### 默认账号

| 用户名 | 密码 | 角色 |
|--------|------|------|
| admin | admin123 | 管理员 |

## 核心功能说明

### 人脸识别流程

1. **人脸检测**: YuNet / YOLOv8 定位人脸区域
2. **特征提取**: HOG（方向梯度直方图）提取人脸特征向量
3. **特征对比**: 计算余弦相似度
4. **阈值判断**: 相似度 >= 阈值 → 已知人员，否则 → 陌生人

```
similarity = cos(θ) = (A·B) / (||A|| × ||B||)
```

### 摔倒检测

使用 YOLOv8-pose 提取 17 个 COCO 关键点，计算肩-髋中轴线与垂直方向的夹角，角度低于 30° 判定为疑似摔倒。

### 情感分析

支持 7 种表情：开心、平静、悲伤、惊讶、生气、恐惧、厌恶

## 数据库模型

| 模型 | 说明 |
|------|------|
| User | 工作人员（用户名、角色、部门） |
| Elderly | 老年人（健康信息、房间号、紧急联系人） |
| Volunteer | 义工（技能、服务时长） |
| Alert | 告警记录（类型、严重程度、处理状态） |
| FaceRecord | 人脸特征库（HOG 编码、关联人员） |
| SystemConfig | 系统配置（键值存储） |
| DetectionConfig | 检测参数（阈值配置） |

## 常见问题

### 摄像头无法启动？
1. 检查摄像头是否被其他程序占用
2. 确保浏览器已授权摄像头权限
3. 尝试刷新页面后重新启动

### 人脸识别不准确？
1. 调整检测参数配置中的阈值
2. 上传更清晰的人脸照片
3. 确保光线充足

### dlib / face-recognition 安装失败？
Windows 下推荐使用 conda 安装：
```bash
conda install -c conda-forge dlib
pip install face-recognition
```

### 重启后数据丢失？
1. 确认数据库文件存在（`instance/elderly_care.db`）
2. 定期使用数据备份功能

## 更新日志

### v1.0.0 (2026-05-18)
- 初始版本发布
- 实现基础 CRUD 功能
- 集成 OpenCV 人脸检测
- 添加情感分析功能
- 实现陌生人识别与摔倒检测
- 告警记录系统
- 数据备份恢复
- 响应式前端界面

## 贡献者

初晓旬依

## 许可证

MIT License

---

**智慧养老系统 - 让科技温暖夕阳红**
