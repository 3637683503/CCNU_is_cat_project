#考虑用遍历来确定可用摄像头
import cv2

def find_available_cameras():
    available_cameras = []
    for i in range(200):  # 假设最多可能有 200 个摄像头
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
        else:
            pass
    return available_cameras

print(find_available_cameras())