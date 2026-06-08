from collections import deque
import platform
import threading
import time
import cv2,gc,ast,os,json,asyncio
import imageio
import torch,torchvision
import numpy as np
from torchvision.transforms import functional as F
import pandas as pd
from tqdm import tqdm
from model.videoInput import labelConfig
from common.config import *
from common.ajaxResult import *
from common.redisManager import RedisManager
from .backbone import resnet50_fpn_backbone
from .network_files import MaskRCNN
from PIL import Image
from torchvision import transforms
# from draw_box_utils import draw_objs


VIDEO_OUT_WIDTH = 640       #输出视频宽度
VIDEO_OUT_HEIGHT = 360      #输出视频高度
VIDEO_OUT_FPS = 5           #输出视频FPS
DEF_MAX_FRAME_NUM = VIDEO_OUT_FPS*DEDUCTION_ORG_VIDEO_TIME_LONG
osName = platform.system().lower()


class FrameBuffer:
    def __init__(self, max_frame_num=DEF_MAX_FRAME_NUM):
        self.buffer = deque(maxlen=max_frame_num)
        self.max = max_frame_num
        self.lock = threading.Lock()  # 线程安全锁

    def add_frame(self, frame):
        """添加新帧到缓冲区"""
        with self.lock:
            timestamp = time.time()
            local_time = time.localtime(timestamp)
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
            # self.buffer.append((frame, formatted_time))
            # 存储原始帧
            self.buffer.append((frame, formatted_time))  # 关键修改：使用frame.copy()

    def clear_buffer(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()

    def save_all_frames(self, outPath):
        """根据开始时间和结束时间，以及上溯和延后帧数，获取特定帧"""
        with self.lock:
            lenthThis: int = len(self.buffer)
            begin_frame_num = 0
            end_frame_num = lenthThis - 1
            try:
                if osName == "windows":
                    with imageio.get_writer(
                        outPath,
                        fps=VIDEO_OUT_FPS,
                        codec="libx264",
                        pixelformat="yuv420p",
                        quality=6,
                        output_params=[
                            '-crf', '30',
                            '-preset', 'fast',
                            '-movflags', '+faststart' # Web优化：立即播放
                        ]
                    ) as writer:
                        # print("outPath:",outPath)
                        for i in range(begin_frame_num, end_frame_num):
                            # print("save_frames_as_video:",i)
                            (frame, formatted_time) = self.buffer[i]
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            writer.append_data(rgb_frame)
                else:
                    with imageio.get_writer(
                            outPath,
                            fps=VIDEO_OUT_FPS,
                            codec="h264_nvenc",
                            pixelformat="yuv420p",
                            output_params=[
                                '-cq', '30',
                                '-preset', 'p1'
                            ]
                    ) as writer:
                        # print("outPath:",outPath)
                        for i in range(begin_frame_num, end_frame_num):
                            # print("save_frames_as_video:",i)
                            (frame, formatted_time) = self.buffer[i]
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            writer.append_data(rgb_frame)
            except Exception as e:
                print(f"保存视频时发生错误：{e}")
                time.sleep(3)
                if os.path.exists(outPath):
                    os.remove(outPath)

def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone(pretrain_path=TRAIN_ORGMODEL_PATH+"/resnet50-0676ba61.pth", trainable_layers=1)
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # polygons = []
    #选择面积最大的轮廓
    main_contour = max(contours, key=cv2.contourArea)
    # tolerance: 简化程度(值越大点越少)
    # tolerance = 0.5
    # epsilon = tolerance * cv2.arcLength(main_contour, True)
    # main_contour = cv2.approxPolyDP(main_contour, epsilon, True)

    # for contour in contours:
    #     if len(contour) >= 3:  # 至少三个点才能形成多边形
    #         polygons.append(contour.flatten().tolist())
    polygon = main_contour.flatten().tolist()
    # print(f"polygon-----{polygon}")
    return polygon


def masks_to_coco_polygon(masks):
    """
    将多个二值掩码转换为 COCO 的 Polygon 格式 segmentation 数据
    :param masks: shape [N, H, W], bool 类型的掩码数组
    :return: list of list, 每个元素是坐标列表组成的多边形
    """
    segmentations = []
    for mask in masks:
        polygon = mask_to_polygon(mask)
        segmentations.append(polygon)
    # print(f"segmentations-----{segmentations}")
    return segmentations

def generate_coco_annotations(results, video_info,image_name,category_classes):
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
                "license":video_info["license"]
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
            "licenses":[{
                "id":1,
                "name":video_info["license"],
                "url":"",
                "file_url":"",
                "version":"1.0"
            }],
            "categories": [{"id": int(k), "name": v,"supercategory":"none"} for k, v in category_classes.items()]
        }

        if results is not None and len(results)>0:
            ii:int=0
            for ann in results:
                ii+=1
                # print(ann['bbox'])
                annbox=ann['bbox']
                annbox = [
                    annbox[0],
                    annbox[1],
                    annbox[2] - annbox[0],
                    annbox[3] - annbox[1]
                ]
                coco_ann = {
                    "id": ii,
                    "image_id": 0,
                    "category_id": ann['labelid'],
                    "segmentation": ann['segmentation'],
                    "bbox": annbox,
                    "score": ann['score'],
                    "area": annbox[2] * annbox[3],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(coco_ann)

        return coco_data

# print(f"处理完成，结果视频保存至：{result_path}")
def labelAction(LabelConfig: labelConfig,redis:RedisManager=None):
    # buffer=FrameBuffer()
    num_classes = 0  # 不包含背景
    box_thresh = MASKRCNN_DEFALT_THRESH # 检测阈值
    weights_path = DEDUCTION_MODEL_PATH+"/"+LabelConfig.id+"_"+LabelConfig.modelType+".pth"
    img_path = "/data/train/data/images/frame_0063.jpg"
    label_json_path = TRAIN_DATASETS_SAVE_PATH+"/"+LabelConfig.id+"_classes.json"

     # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r',encoding="utf-8") as json_file:
        category_index:dict = json.load(json_file)
    num_classes=len(category_index)

    # get devices
    device = torch.device(MASKRCNN_DEFALT_DEVICE if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    model.eval()  # 进入验证模式

    
    """执行视频推理"""
    # 初始化视频读写器
    # cv2
    video_self_fps:int=DEDUCTION_VIDEO_FPS
    video_cute_fps:int=DEDUCTION_VIDEO_CUTE_FPS
    video_width:int = DEDUCTION_VIDEO_WIDTH
    video_height:int = DEDUCTION_VIDEO_HEIGHT
    max_frames:int=DEDUCTION_VIDEO_FPS*DEDUCTION_ORG_VIDEO_TIME_LONG  #视频文件也最多截取前5分钟
    frame_count:int=0

    video_info = {
        'fps': video_self_fps,
        'width': video_width,
        'height': video_height,
        'license':DEDUCTION_LICENSE_NAME
    }

    cap=cv2.VideoCapture(LabelConfig.videoFile,cv2.CAP_FFMPEG) if LabelConfig.videoType=="1" else cv2.VideoCapture(DEDUCTION_ORG_VIDEO_PATH+"/"+LabelConfig.videoFile)
    if LabelConfig.videoType=="2":
        #视频文件处理
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames=min(max_frames,total_frames)

    with tqdm(total=max_frames, desc=f"Processing first {max_frames} frames") as pbar:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break
            if redis and redis.get(f"{START_VIDEO_PRE_MARK_PX}{LabelConfig.id}")=="end":
                print("视频预标注任务结束")
                break
            # 计算时间戳
            # timestamp = frame_count / video_self_fps
            if frame_count % video_cute_fps == 0:
                print(f"处理第{frame_count}帧")
                # 预处理
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                # pil_image = Image.open(img_path).convert("RGB")
                # from pil image to tensor, do not normalize image
                data_transform = transforms.Compose([transforms.ToTensor()])
                img = data_transform(pil_image)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)
                # model.eval()
                # 模型推理
                with torch.no_grad():
                    img_height, img_width = img.shape[-2:]
                    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                    model(init_img)
                    t_start = time_synchronized()
                    prediction = model(img.to(device))[0]
                    t_end = time_synchronized()
                    print("inference+NMS time: {}".format(t_end - t_start))
                    # prediction=predictions[0]
                    predict_boxes = prediction["boxes"].to("cpu").detach().numpy()
                    predict_classes = prediction["labels"].to("cpu").detach().numpy()
                    predict_scores = prediction["scores"].to("cpu").detach().numpy()
                    predict_mask_org = prediction["masks"].cpu().detach().numpy()
                    predict_mask = np.squeeze(predict_mask_org, axis=1)  # [batch, 1, h, w] -> [batch, h, w]
                    # predict_mask = np.where(predict_mask > 0.7, True, False)
                    # print("predict_mask:",predict_mask)
                    if len(predict_boxes) == 0:
                        print("没有检测到任何目标!")
                        if LabelConfig.isGetNoLabel:
                            frame_save_name=f"{LabelConfig.guid}_frames_{frame_count:05d}.jpg"
                            frame_save_path=f"{DEDUCTION_FM_OUTPUT_IMAGE_PATH}/{frame_save_name}"
                            pil_image.save(frame_save_path)
                            coco_data=generate_coco_annotations(None,video_info,frame_save_name,category_index)
                            threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH+SEND_LABEL_TO_JAVA,{
                                "guId":LabelConfig.guid,
                                "imageName":frame_save_name,
                                "cocoJsonStr":json.dumps(coco_data,ensure_ascii=False)
                            })).start()
                    else:
                        print(f"检测阈值{box_thresh},检测到{len(predict_boxes)}个目标!")
                        maskss = np.where(predict_mask > box_thresh, True, False)
                        segmentations=masks_to_coco_polygon(maskss)
                        results = []
                        for i in range(len(predict_scores)):
                            # 解析检测信息
                            label_id = str(predict_classes[i])
                            box = predict_boxes[i].astype(int)
                            score = predict_scores[i]
                            label = category_index.get(label_id, 'N/A')

                            # 记录结果数据
                            results.append({
                                "label": label,
                                "labelid": label_id,
                                "score": float(score),
                                "bbox": box.tolist(),
                                "segmentation": [segmentations[i]]
                            })
                        # print(f"检测结果：{results}")
                        if results is not None and len(results)>0:
                            frame_save_name=f"{LabelConfig.guid}_frames_{frame_count:05d}.jpg"
                            frame_save_path=f"{DEDUCTION_FM_OUTPUT_IMAGE_PATH}/{frame_save_name}"
                            pil_image.save(frame_save_path)
                            coco_data=generate_coco_annotations(results,video_info,frame_save_name,category_index)
                            #正式需要推到Java端
                            # print(frame_save_path)
                            # print(json.dumps(coco_data,ensure_ascii=False))
                            # asyncio.create_task(postAsyncJsonWithOutJwt)
                            threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH+SEND_LABEL_TO_JAVA,{
                                "guId":LabelConfig.guid,
                                "imageName":frame_save_name,
                                "cocoJsonStr":json.dumps(coco_data,ensure_ascii=False)
                            })).start()
                # framesize = cv2.resize(frame, (VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT))
                # buffer.add_frame(framesize)
            frame_count += 1
        pbar.update(1)
    # 释放资源
    cap.release()
    """
    outPath=DEDUCTION_ORG_VIDEO_PATH+"/"+LabelConfig.guid+".mp4"
    buffer.save_all_frames(outPath)
    time.sleep(10)
    buffer.clear_buffer()
    """
    print(f"处理完成")
