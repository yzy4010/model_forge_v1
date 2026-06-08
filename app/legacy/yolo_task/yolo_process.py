import threading
import uuid
from datetime import datetime

import cv2
import os
import json
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from skimage import measure
from common.config import *
from common.redisManager import RedisManager
from model.videoInput import labelConfig
from common.ajaxResult import *


class VideoProcessor:
    def __init__(self, model_path):

        # processor_model = DEDUCTION_MODEL_PATH + '/' + model_path
        processor_model = DEDUCTION_MODEL_PATH + '/' + model_path + '.pt'
        print(processor_model)
        if not os.path.isfile(processor_model):
            print(f"文件 {processor_model} 不存在")
            return
        # if not os.path.exists(processor_model):
        #     print(f"文件夹不存在: {processor_model}_加载初始模型")
        #     return
        self.model = YOLO(processor_model)
        self.model.names.items()  # 获取模型标签
        self.annotation_id = 1
        self.coco_data = {
            "info": [],
            "images": [],
            "annotations": [],
            "licenses": [],
            "categories": []
        }

    def init_categories(self):
        self.coco_data["categories"] = [
            {"id": k, "name": v, "supercategory": "object"}
            for k, v in self.model.names.items()
        ]

    def process_video(self, labelConfig: labelConfig, redis: RedisManager):

        video_self_fps: int = DEDUCTION_VIDEO_FPS  # 推理时摄像头原始视频帧率
        video_cute_fps: int = DEDUCTION_VIDEO_CUTE_FPS  # 推理时摄像头截取视频帧率
        video_width: int = DEDUCTION_VIDEO_WIDTH  # 推理时摄像头原始视频宽度
        video_height: int = DEDUCTION_VIDEO_HEIGHT  # 推理时摄像头截取视频高度

        save_video_width: int = 640  # 保存视频的指定宽
        save_video_height: int = 360  # 保存视频的指定高
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
        self.init_categories()

        """
        # 新增：创建视频保存目录
        # 获取当前日期时间 格式化为yyyymmdd字符串
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y%m%d")
        # 保存的视频文件名称
        video_id = uuid.uuid4().hex
        video_name = f"{video_id}.mp4"
        # 保存的目标路径以及文件名
        video_save_dir = DEDUCTION_ORG_VIDEO_PATH + '/' + formatted_date + '/' + video_name
        os.makedirs(os.path.dirname(video_save_dir), exist_ok=True)
        # os.makedirs(video_save_dir, exist_ok=True)
        # video_path = os.path.join(video_save_dir, video_name)
        """
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

        # 初始化视频写入器
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #  mp4v 保存的视频无法在浏览器播放
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # writer = cv2.VideoWriter(video_save_dir, fourcc, video_self_fps, (video_width, video_height))
        # writer = cv2.VideoWriter(video_save_dir, fourcc, video_self_fps, (save_video_width, save_video_height))

        if not cap.isOpened():
            print("无法打开视频源")
            return

        if labelConfig.videoType == "2":
            # 视频文件处理
            # 视频文件最大贞数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = min(max_frames, total_frames)

        frame_count = 0
        save_count = 0
        # save_video_flag = False  # 新增：标记是否需要保存视频
        with tqdm(total=max_frames, desc=f"Processing first {max_frames} frames") as pbar:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取视频帧")
                    break
                if redis.get(f"{START_VIDEO_PRE_MARK_PX}{labelConfig.id}") == "end":
                    print("视频预标注任务结束")
                    # 设置保存视频标志
                    # save_video_flag = True
                    break

                # # 新增：视频流实时保存
                # if labelConfig.videoType == "1" and writer is not None:
                #     writer.write(frame)
                # 调整帧大小到自定义分辨率
                # resized_frame = cv2.resize(frame, (video_width, video_height))
                # resized_frame = cv2.resize(frame, (save_video_width, save_video_height))
                # writer.write(resized_frame)
                # 对于所有视频类型，写入调整大小后的帧
                # if writer is not None:
                #     writer.write(resized_frame)

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

                    # 进行推理
                    results = self.model(frame, conf=YOLOV8_DEFALT_THRESH)
                    # results = self.model(frame, imgsz=640, conf=0.5, verbose=False)
                    # self.process_segmentation(results[0], save_count)
                    coco_data = self.generate_coco_annotations(results[0], video_info, frame_save_name,
                                                               labelConfig.isGetNoLabel)
                    # print(frame_save_path)
                    # print(coco_data)
                    if coco_data is not None:
                        # save_coco_json_1("/data/json/" + frame_save_name.replace(".jpg","") + ".json", coco_data)
                        threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_LABEL_TO_JAVA, {
                            "guId": labelConfig.guid,
                            "imageName": frame_save_name,
                            "cocoJsonStr": json.dumps(coco_data, ensure_ascii=False)
                        })).start()
                    # 传入java逻辑处理

                    save_count += 1

                frame_count += 1
            pbar.update(1)

        cap.release()
        # return self.coco_data
        # 只在Redis标记为"end"时保存视频
        """
        if save_video_flag:
            if writer is not None:
                writer.release()
                threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + POST_DEDUCTION_SAVE_VIDEO_JAVA, {
                    "videoName": video_name,
                    "videoUrl": formatted_date + '/' + video_name,
                })).start()
            print(f"已保存至: {video_save_dir}")
        """

    # json 内容
    def generate_coco_annotations(self, result, video_info, image_name, isGetNoLabel: bool = False):
        """生成COCO格式标注"""
        # categories=[]
        # for k, v in self.label_map.items():
        #     # if v not in results: continue
        #     categories.append({"id": v, "name": k})
        coco_data = {
            "info": {
                "description": "Video Inference Results",
                "fps": video_info["fps"],
                "width": video_info["width"],
                "height": video_info["height"],
                "license": video_info["license"]
            },
            "images": [{
                "id": 0,
                "width": video_info["width"],
                "height": video_info["height"],
                "file_name": image_name,
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
            # "categories": [{"id": v, "name": k} for k, v in self.model.names.items()]
            "categories": [{"id": k, "name": v, "supercategory": "object"} for k, v in self.model.names.items()]
        }

        annotation_id = 0
        if result.masks is None:
            if isGetNoLabel:
                return coco_data
            else:
                return None

            # 处理每个检测结果
        for i, (box, mask, score, cls_id) in enumerate(zip(
                result.boxes.xyxy,
                result.masks.xy,
                result.boxes.conf,
                result.boxes.cls
        )):
            # 获取基础信息
            cls_id = int(cls_id)
            # 转换为COCO格式的bbox [x,y,width,height]
            bbox = [
                int(float(box[0])),
                int(float(box[1])),
                int(float(box[2] - box[0])),
                int(float(box[3] - box[1]))
            ]

            # 处理分割掩码（多边形）
            segmentation = []
            for point in mask:
                segmentation.extend([int(float(point[0])), int(float(point[1]))])

            # 计算面积
            area = int(box[2] * box[3])

            # 创建COCO标注
            annotation = {
                "id": annotation_id,
                "image_id": 0,
                "category_id": cls_id,
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "score": float(score)
            }

            coco_data["annotations"].append(annotation)
            # self.annotation_id += 1
            annotation_id += 1
        return coco_data

    # region 未使用到
    # 成功处理标记和分割掩码
    def process_segmentation(self, result, image_id):

        # 如果没有掩码数据，则跳过该结果
        if result.masks is None:
            print("没有掩码数据")
            return

            # 处理每个检测结果
        for i, (box, mask, score, cls_id) in enumerate(zip(
                result.boxes.xyxy,
                result.masks.xy,
                result.boxes.conf,
                result.boxes.cls
        )):
            # 获取基础信息
            cls_id = int(cls_id)
            # 转换为COCO格式的bbox [x,y,width,height]
            bbox = [
                float(box[0]),
                float(box[1]),
                float(box[2] - box[0]),
                float(box[3] - box[1])
            ]

            # 处理分割掩码（多边形）
            segmentation = []
            for point in mask:
                segmentation.extend([float(point[0]), float(point[1])])

            # 计算面积
            area = float(box[2] - box[0]) * float(box[3] - box[1])

            # 创建COCO标注
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": cls_id,
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "score": float(score)
            }

            self.coco_data["annotations"].append(annotation)
            self.annotation_id += 1

    def mask_to_polygon(self, mask):
        """将二值掩码转换为COCO多边形格式"""
        contours = measure.find_contours(mask, YOLOV8_DEFALT_THRESH)
        polygons = []
        for contour in contours:
            contour = np.flip(contour, axis=1)  # 转换坐标系
            segmentation = contour.ravel().tolist()
            if len(segmentation) >= 6:  # 至少需要3个点(6个坐标值)
                polygons.append(segmentation)
        return polygons

    def process_segmentation_1(self, result, image_id, frame):
        # 获取分割结果
        masks = result.masks
        boxes = result.boxes

        if masks is None:
            return

        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            # 获取基础信息
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            xyxy = box.xyxy.cpu().numpy().squeeze()

            # 转换掩码
            bin_mask = mask.data.cpu().numpy().squeeze()
            polygons = self.mask_to_polygon(bin_mask)

            # 计算区域面积
            area = np.sum(bin_mask)  # 像素数量
            # 创建COCO标注
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": cls_id,
                "segmentation": polygons,
                "area": float(area),
                "bbox": [float(xyxy[0]), float(xyxy[1]),
                         float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])],
                "iscrowd": 0,
                "score": conf
            }

            self.coco_data["annotations"].append(annotation)
            self.annotation_id += 1

    def process_results(self, result, image_id, timestamp):
        boxes = result.boxes.cpu()
        for box in boxes:
            xyxy = box.xyxy.numpy().squeeze().tolist()
            conf = box.conf.item()
            cls_id = int(box.cls.item())

            # 转换bbox为COCO格式 (x, y, w, h)
            x_min, y_min, x_max, y_max = xyxy
            width = x_max - x_min
            height = y_max - y_min

            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "score": conf,
                "iscrowd": 0,
                "timestamp": round(timestamp, 3),
            }

            self.coco_data["annotations"].append(annotation)
            self.annotation_id += 1

    # endregion

    # 保存COCO数据
    def save_coco_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, ensure_ascii=False, indent=2)


def start_process_video(labelConfig: labelConfig, redis: RedisManager):
    model_name = f"{labelConfig.id}_{labelConfig.modelType}"
    print(model_name)
    # 使用示例
    processor = VideoProcessor(model_name)
    # 处理视频文件
    processor.process_video(labelConfig, redis)


def save_coco_json_1(output_path, coco_data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
# if __name__ == "__main__":
#     class labelConfig():
#         # 动作模型id
#         id: str = '2'
#         # 动作模型算法类型  1：Mask R-CNN  2：yolov8  3：YoLact  4：Torch.nn
#         modelType: str = '2'
#         # 视频流或者视频文件地址
#         videoFile: str = ''
#         # 视频文件的类型 1 视频流 2 视频文件地址
#         videoType: str = '2'
#
#
#     model_name = f"{labelConfig.id}_{labelConfig.modelType}"
#     print(model_name)
#     # 使用示例
#     processor = VideoProcessor(model_name)
#     # 处理视频文件
#     coco_data = processor.process_video(labelConfig)
#
#     # 保存COCO格式结果
#     # processor.save_coco_json("D:/WuJianPing/GR/json/output.json")
#
#     print("处理完成，结果已保存到 output.json")
