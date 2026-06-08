import threading
import time
import cv2,gc,ast,os,json,asyncio
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




def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone(pretrain_path=TRAIN_ORGMODEL_PATH+"/resnet50-0676ba61.pth", trainable_layers=3)
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
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # 至少三个点才能形成多边形
            polygons.append(contour.flatten().tolist())
    return polygons


def masks_to_coco_polygon(masks):
    """
    将多个二值掩码转换为 COCO 的 Polygon 格式 segmentation 数据
    :param masks: shape [N, H, W], bool 类型的掩码数组
    :return: list of list, 每个元素是坐标列表组成的多边形
    """
    segmentations = []
    for mask in masks:
        polygons = mask_to_polygon(mask)
        segmentations.append(polygons)
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
                "id": 1,
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

        ii:int=0
        for ann in results:
            ii+=1
            # print(ann['bbox'])
            annbox=ann['bbox']
            coco_ann = {
                "id": ii,
                "image_id": 1,
                "category_id": ann['labelid'],
                "segmentation": ann['segmentation'],
                "bbox": ann['bbox'],
                "score": ann['score'],
                "area": (annbox[2] - annbox[0]) * (annbox[3] - annbox[1]),
                "iscrowd": 0
            }
            coco_data["annotations"].append(coco_ann)

        return coco_data

class VideoInference:
    def __init__(self, model_path,labelJson_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        """从模型结构中提取原始标签映射"""
        if hasattr(self.model, 'label_map'):
            self.label_map = self.model.label_map
            print(f"从模型加载标签映射: {self.label_map}")
        else:
            print("警告：模型不包含标签映射信息")
        # self.label_map = label_map
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # 可视化配置
        
        self.conf_threshold = 0.2   # 置信度阈值，正式上线这个值至少要达到0.8

        # self.colors = np.random.randint(0, 255, (100, 3))  # 每个类别随机颜色
        # self.mask_alpha = 0.3  # 掩码透明度

    def load_model(self, model_path):
        """加载完整模型文件"""
        model = torch.load(model_path, map_location="cpu",weights_only=False)
        model.eval()
        return model.to(self.device)

    def preprocess_frame(self, frame):
        """预处理视频帧"""
        # 保持原始尺寸，仅进行归一化
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(frame_rgb).unsqueeze(0)
        return tensor.to(self.device)

    def process_detections(self, frame, detections, timestamp):
        """处理并可视化检测结果"""
        height, width = frame.shape[:2]
        results = []
        # print(detections['labels'])
        
        # 提取预测结果
        boxes = detections['boxes'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        masks = detections['masks'].cpu().numpy()
        # print(f"labels:-------{labels}")
        # print(f"scores:-------{scores}")

        for i in range(len(scores)):
            if scores[i] >= self.conf_threshold:
                # 解析检测信息
                label_id = labels[i]
                box = boxes[i].astype(int)
                # print(masks[i])
                mask = masks[i][0]  # 去除通道维度
                score = scores[i]
                label = self.inverse_label_map.get(label_id, 'unknown')

                # 记录结果数据
                results.append({
                    "timestamp": round(timestamp, 3),
                    "label": label,
                    "score": float(score),
                    "bbox": box.tolist(),
                    "segmentation": self.mask_to_polygon(mask)
                })
        return results

    def mask_to_polygon(self, mask):
        """将掩码转换为多边形坐标"""
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            if contour.shape[0] < 4:  # 至少需要3个点组成多边形
                continue
            contour = contour.squeeze().astype(float)
            # 简化多边形点数
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append(approx.flatten().tolist())
        
        return polygons

    def inference_video(self, LabelConfig: labelConfig,redis:RedisManager):
        """执行视频推理"""
        # 初始化视频读写器
        # cv2
        video_self_fps:int=DEDUCTION_VIDEO_FPS
        video_cute_fps:int=DEDUCTION_VIDEO_CUTE_FPS
        video_width:int = DEDUCTION_VIDEO_WIDTH
        video_height:int = DEDUCTION_VIDEO_HEIGHT
        max_frames:int=DEDUCTION_VIDEO_FPS*DEDUCTION_ORG_VIDEO_TIME_LONG  #视频文件也最多截取前10分钟
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
                if redis.get(f"{START_VIDEO_PRE_MARK_PX}{LabelConfig.id}")=="end":
                    print("视频预标注任务结束")
                    break
                # 计算时间戳
                timestamp = frame_count / video_self_fps
                if frame_count % video_cute_fps == 0:
                    print(f"处理第{frame_count}帧")
                    
                    # 预处理
                    input_tensor = self.preprocess_frame(frame)
                    
                    # 模型推理
                    with torch.no_grad():
                        detections = self.model(input_tensor)[0]
                    
                    # print(detections)
                    # 后处理与可视化
                    frame_results = self.process_detections(
                        frame, detections, timestamp
                    )
                    if frame_results is not None and len(frame_results)>0:
                        #将processed_frame保存为图片，frame_results输出为coco_json格式
                        frame_save_name=f"{LabelConfig.guid}_frames_{frame_count:05d}.jpg"
                        frame_save_path=f"{DEDUCTION_FM_OUTPUT_IMAGE_PATH}/{frame_save_name}"
                        cv2.imwrite(frame_save_path, frame)
                        coco_data=self.generate_coco_annotations(frame_results,video_info,frame_save_name)
                        #正式需要推到Java端
                        # print(frame_save_path)
                        # print(json.dumps(coco_data,ensure_ascii=False))
                        threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH+SEND_LABEL_TO_JAVA,{
                                "guId":LabelConfig.guid,
                                "imageName":frame_save_name,
                                "cocoJsonStr":json.dumps(coco_data,ensure_ascii=False)
                            })).start()
                frame_count += 1
                pbar.update(1)
        # 释放资源
        cap.release()       
        print(f"处理完成")
    def generate_coco_annotations(self, results, video_info,image_name):
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
                "id": 1,
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
            "categories": [{"id": v, "name": k} for k, v in self.label_map.items()]
        }

        ii:int=0
        for ann in results:
            ii+=1
            # print(ann['bbox'])
            annbox=ann['bbox']
            coco_ann = {
                "id": ii,
                "image_id": 1,
                "category_id": self.label_map[ann['label']],
                "segmentation": ann['segmentation'],
                "bbox": ann['bbox'],
                "score": ann['score'],
                "area": (annbox[2] - annbox[0]) * (annbox[3] - annbox[1]),
                "iscrowd": 0
            }
            coco_data["annotations"].append(coco_ann)

        return coco_data


# print(f"处理完成，结果视频保存至：{result_path}")
def labelAction(LabelConfig: labelConfig,redis:RedisManager=None):
    num_classes = 0  # 不包含背景
    box_thresh = MASKRCNN_DEFALT_THRESH # 检测阈值
    weights_path = DEDUCTION_MODEL_PATH+"/"+LabelConfig.id+"_"+LabelConfig.modelType+".pth"
    img_path = "/data/train/data/images/frame_0063.jpg"
    label_json_path = DEDUCTION_MODEL_PATH+"/"+LabelConfig.id+"_"+LabelConfig.modelType+"_classes.json"

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
    max_frames:int=DEDUCTION_VIDEO_FPS*DEDUCTION_ORG_VIDEO_TIME_LONG  #视频文件也最多截取前10分钟
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
                    predict_boxes = prediction["boxes"].to("cpu").numpy()
                    predict_classes = prediction["labels"].to("cpu").numpy()
                    predict_scores = prediction["scores"].to("cpu").numpy()
                    predict_mask_org = prediction["masks"].cpu().numpy()
                    predict_mask = np.squeeze(predict_mask_org, axis=1)  # [batch, 1, h, w] -> [batch, h, w]
                    # predict_mask = np.where(predict_mask > 0.7, True, False)
                    # print("predict_mask:",predict_mask)
                    if len(predict_boxes) == 0:
                        print("没有检测到任何目标!")
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
                                "segmentation": segmentations[i]
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
            frame_count += 1
            pbar.update(1)
    # 释放资源
    cap.release()       
    print(f"处理完成")








    # 初始化推理器
    inferencer = VideoInference(model_path=DEDUCTION_MODEL_PATH+"/"+LabelConfig.id+"_"+LabelConfig.modelType+".pth",
        device=MASKRCNN_DEFALT_DEVICE)
    result_path = inferencer.inference_video(LabelConfig,redis)
