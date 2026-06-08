import threading
import cv2
import os
import json
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from skimage import measure
from ..common.config import *
from ..common.redisManager import RedisManager
from model.videoInput import labelConfig
from ..common.ajaxResult import *


class VideoProcessor:
    def __init__(self):

        self.annotation_id = 1
        self.coco_data = {
            "info": [],
            "images": [],
            "annotations": [],
            "licenses": [],
            "categories": [{"id": 1, "name": "N/A", "supercategory": "object"}]
        }

    def process_video(self, labelConfig: labelConfig, redis: RedisManager):

        video_self_fps: int = DEDUCTION_VIDEO_FPS  # 推理时摄像头原始视频帧率
        video_cute_fps: int = DEDUCTION_VIDEO_CUTE_FPS  # 推理时摄像头截取视频帧率
        video_width: int = DEDUCTION_VIDEO_WIDTH  # 推理时摄像头原始视频宽度
        video_height: int = DEDUCTION_VIDEO_HEIGHT  # 推理时摄像头截取视频高度
        # 视频流最大贞数
        max_frames: int = DEDUCTION_VIDEO_FPS * DEDUCTION_ORG_VIDEO_TIME_LONG  # 视频文件也最多截取前10分钟

        video_info = {
            'fps': video_self_fps,
            'width': video_width,
            'height': video_height,
            'license': DEDUCTION_LICENSE_NAME
        }
        self.coco_data["info"].append(video_info)

        licenses_info = {
            'id': 1,
            'name': video_info["license"],
            'url': "",
            'file_url': "",
            'version': "1.0"
        }
        self.coco_data["licenses"].append(licenses_info)

        # 判断视频类型选择那个方法 1 视频流  否则视频文件
        if labelConfig.videoType == "1":
            cap = cv2.VideoCapture(labelConfig.videoFile)
            # 获取摄像头参数
            # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            cap = cv2.VideoCapture(DEDUCTION_ORG_VIDEO_PATH + "/" + labelConfig.videoFile)
            # 判断视频流或者视频文件是否打开
            # if not cap.isOpened():
            #     raise ValueError("无法打开视频源")

        if labelConfig.videoType == "2":
            # 视频文件处理
            # 视频文件最大贞数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = min(max_frames, total_frames)

        frame_count = 0
        save_count = 0

        with tqdm(total=max_frames, desc=f"Processing first {max_frames} frames") as pbar:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取视频帧")
                    break
                if redis.get(f"{START_VIDEO_PRE_MARK_PX}0000") == "end":
                    print("视频预标注任务结束")
                    break

                # 计算时间戳
                timestamp = frame_count / video_self_fps
                if frame_count % video_cute_fps == 0:
                    print(f"处理第{frame_count}帧")
                    frame_save_name = f"{labelConfig.guid}_frames_{frame_count:05d}.jpg"
                    frame_save_path = f"{DEDUCTION_FM_OUTPUT_IMAGE_PATH}/{frame_save_name}"
                    # path = 'D:/WuJianPing/GR/images'
                    # frame_save_path = f"{path}/{frame_save_name}"
                    cv2.imwrite(frame_save_path, frame)

                    # 添加到COCO images
                    image_info = {
                        "id": save_count,
                        "file_name": frame_save_name,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "license": 1,
                        "date_captured": ""
                    }
                    self.coco_data["images"].append(image_info)

                    coco_data = {
                        "info": {
                            "description": "Video Inference Results",
                            "fps": video_info["fps"],
                            "width": video_info["width"],
                            "height": video_info["height"],
                            "license": video_info["license"]
                        },
                        "images": [{
                            "id": 1,
                            "width": video_info["width"],
                            "height": video_info["height"],
                            "file_name": frame_save_name,
                            "license": 1,
                            "date_captured": ""
                        }],
                        "annotations": [],
                        "licenses": [{
                            "id": 1,
                            "name": video_info["license"],
                            "url": "",
                            "file_url": "",
                            "version": "1.0"
                        }],
                        "categories": [{"id": 1, "name": "N/A", "supercategory": "object"}]
                    }

                    threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_LABEL_TO_JAVA, {
                            "guId": labelConfig.guid,
                            "imageName": frame_save_name,
                            "cocoJsonStr": json.dumps(coco_data, ensure_ascii=False)
                        })).start()

                    save_count += 1
                frame_count += 1
            pbar.update(1)

        cap.release()
        # return self.coco_data

    # endregion


def start_empty_video(labelConfig: labelConfig, redis: RedisManager):
    model_name = f"{labelConfig.id}_{labelConfig.modelType}"
    print(model_name)
    # 使用示例
    processor = VideoProcessor()
    # 处理视频文件
    processor.process_video(labelConfig, redis)

