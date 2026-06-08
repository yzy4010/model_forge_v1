import cv2
import os
import time
import json
import csv
import re

def extract_frames(video_path, output_dir, fps=6, target_width=1920, target_height=1080):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frame_delay = 1 / fps  # 计算每帧之间的时间间隔
    last_time = time.time()
    
    while success:
        # current_time = time.time()
        # if current_time - last_time >= frame_delay:
        #     last_time = current_time
        if count%fps==0:
            #cv2.imwrite(frame_path, image)    -------原始大小
            resized_image = cv2.resize(image, (target_width, target_height))
            frame_path = os.path.join(output_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(frame_path, resized_image)
        count += 1
        success, image = vidcap.read()

# extract_frames('VID_20240425_110744.mp4', 'output_frames')

# 输入输出路径
coco_json_path = 'output_frames/visionai-newest.json'
output_csv_path = 'output_frames/video_full_annotations_new.csv'

# 加载COCO数据
with open(coco_json_path, 'r',encoding='utf-8') as f:
    coco_data = json.load(f)

# 创建映射字典
image_id_to_info = {img['id']: img for img in coco_data['images']}
category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

# 准备写入CSV
with open(output_csv_path, 'w', newline='',encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 合并表头
    writer.writerow(['filename',
        'timestamp', 'label', 
        'xmin', 'ymin', 'xmax', 'ymax',  # 边界框坐标
        'segmentation'                    # 分割多边形
    ])
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        img_info = image_id_to_info.get(image_id)
        if not img_info:
            continue
        
        # ========== 通用处理 ==========
        # 提取帧数和时间戳
        filename = img_info['file_name']
        frame_match = re.search(r'frame_(\d+)\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
        if not frame_match:
            print(f"跳过无法解析帧数的文件: {filename}")
            continue
        frame_number = int(frame_match.group(1))
        timestamp = round(frame_number / 6.0, 2)  # 6帧/秒
        
        # 获取类别
        label = category_id_to_name.get(ann['category_id'], 'unknown')
        
        # ========== BBOX处理 ==========
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
        xmin = round(x, 1)       # 保留1位小数
        ymin = round(y, 1)
        xmax = round(x + w, 1)
        ymax = round(y + h, 1)
        
        # 处理segmentation（取第一个多边形）
        seg_str = ''
        if ann['segmentation']:
            polygon = ann['segmentation'][0]
            # print(type(polygon))
            if not isinstance(polygon,list):
                polygon=ann['segmentation']
            points = [f"{round(polygon[i],1)},{round(polygon[i+1],1)}" for i in range(0, len(polygon), 2)]
            seg_str = ';'.join(points)
    
        # ========== 写入结果 ==========
        writer.writerow([filename,
            timestamp, label,
            xmin, ymin, xmax, ymax,
            seg_str
        ])

print("转换完成！")