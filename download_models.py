"""
模型下载脚本
YOLOv8 模型由 ultralytics 自动下载，此脚本下载 YuNet 人脸检测模型
"""
import os
import urllib.request
import sys

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"已存在: {os.path.basename(dest)}")
        return True
    print(f"下载中: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"完成: {os.path.basename(dest)}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def main():
    print("模型下载工具")
    print("=" * 50)

    # YOLOv8 模型会由 ultralytics 自动下载，只需确保安装正确
    print("\nYOLOv8 模型（自动下载）:")
    print("  - yolov8n.pt (人物检测)")
    print("  - yolov8n-pose.pt (姿态估计)")
    print("  这些模型将在首次使用时由 ultralytics 自动下载。")

    print("\nYuNet 人脸检测模型:")
    for filename, url in MODELS.items():
        dest = os.path.join(BASE_DIR, filename)
        download_file(url, dest)

    print("\n" + "=" * 50)
    print("模型准备完成！")
    print("运行 python run.py 启动系统。")


if __name__ == "__main__":
    main()
