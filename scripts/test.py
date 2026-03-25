import cv2
from ultralytics import YOLO

# 1. 加载模型
# 如果你使用的是 yolov10n.pt，它是最适合 CPU 的轻量化版本
# model = YOLO('D:/projects/yolov10n.pt')
# model = YOLO('D:/projects/model_forge_v1/outputs/cc637265db1c4b708d6f734b822668b2/artifacts/best.pt')
# model = YOLO('D:/projects/best_mobile.pt')
model = YOLO('D:/projects/model_forge_v1/outputs/9ec8d5ae4b6a48999a1385a84746a756/artifacts/best.pt')

# 2. 打开视频文件或摄像头
video_path = 'D:/WorkSpace_test/6474392-uhd_3840_2160_25fps.mp4'  # 如果是摄像头请改为 0
cap = cv2.VideoCapture(video_path)

print(model.names)

if not cap.isOpened():
    print("错误：无法打开视频源。请检查文件名或摄像头连接。")
    exit()

# 3. 创建可调大小的窗口
window_name = "YOLOv10_Office_Monitor"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # 设置一个初始的舒适尺寸

print("--- 推理已启动 ---")
print("如何退出：请先点击一下视频窗口，然后按下键盘上的 'Q' 键或 'Esc' 键")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("视频播放完毕或流中断。")
        break

    frame_count += 1

    # 性能优化：每隔 2 帧处理一次 (跳帧)，避免 CPU 堆积导致延迟
    if frame_count % 2 == 0:
        # device='cpu': 明确指定 CPU
        # imgsz=320: 缩小推理尺寸提升速度
        results = model.predict(source=frame, conf=0.25, device='cpu', imgsz=320, verbose=False)

        # 渲染检测结果
        annotated_frame = results[0].plot()

        # 业务逻辑：检测到手机时在屏幕显示大红字
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            if label == 'cell phone':
                cv2.putText(annotated_frame, "WARNING: PHONE DETECTED", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    else:
        # 非推理帧直接显示上一帧的结果，或者直接显示原图以保持流畅
        annotated_frame = frame

        # 4. 显示画面
    cv2.imshow(window_name, annotated_frame)

    # 5. 退出逻辑
    # waitKey(1) 意味着等待 1 毫秒
    # 0xFF == ord('q') 捕捉键盘上的 'Q' 键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:  # 27 是 Esc 键的 ASCII 码
        print("接收到退出指令，程序正在关闭...")
        break

# 6. 释放资源
cap.release()
cv2.destroyAllWindows()
print("程序已安全退出。")