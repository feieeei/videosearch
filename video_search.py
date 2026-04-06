import cv2
import os
from ultralytics import YOLO
VIDEO_PATH = "test.mp4"    # 视频路径
TARGET_CLASS_NAME = "cell phone" # 检索的物体 
OUTPUT_DIR = "search_results"  
COOLDOWN_SECONDS = 3   # 同一个物体隔3秒截一次图
CONFIDENCE_THRESHOLD = 0.5 # 置信度阈值
def main():
    if not os.path.exists(OUTPUT_DIR):#初始化
        os.makedirs(OUTPUT_DIR)
    #report_path = os.path.join(OUTPUT_DIR, "retrieval_report.txt")
   # report_file = open(report_path, "w", encoding="utf-8")
    model = YOLO("yolov8s.pt") 
    names_dict = model.names
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误 无法打开视频文件: {VIDEO_PATH}，请检查路径。")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) # 获取视频帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 将秒数转换为帧数
    cooldown_frames = int(COOLDOWN_SECONDS * fps)
    last_saved_frame = -cooldown_frames # 初始设为负数
    print(f"目标: [{TARGET_CLASS_NAME}], 视频帧率: {fps:.1f}, 预计总帧数: {total_frames}")
    current_frame = 0
    while cap.isOpened():#读取视频
        ret, frame = cap.read()
        if not ret:
            break 
        current_frame += 1
        # 每3帧分析1次
        if current_frame % 3 != 0:
            continue
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        # 遍历当前帧检测到的所有物体
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = names_dict[class_id]
            conf = float(box.conf[0].item())
            if class_name == TARGET_CLASS_NAME:
                if (current_frame - last_saved_frame) > cooldown_frames:#判断冷却时间是否已过  
                    seconds = int(current_frame / fps)
                    mins, secs = divmod(seconds, 60)
                    time_stamp = f"{mins:02d}:{secs:02d}"
                    print(f"视频时间 {time_stamp}  置信度: {conf:.2f}")
                    # 在画面上画好红框和标签
                    annotated_frame = results[0].plot()
                    # 保存现场图片
                    img_name = f"T_{mins:02d}_{secs:02d}_{TARGET_CLASS_NAME}_{conf:.2f}.jpg"
                    img_path = os.path.join(OUTPUT_DIR, img_name)
                    cv2.imwrite(img_path, annotated_frame)
                    # 写入报告
                   # report_file.write(f"时间段: [{time_stamp}] | 确认度: {conf:.2f}")
                    # 更新记录，重置冷却计时
                    last_saved_frame = current_frame
    cap.release()
    #report_file.close()
    print(f"请打开 '{OUTPUT_DIR}' 文件夹查看自动生成的报告和截图证据。")
if __name__ == "__main__":
    main()