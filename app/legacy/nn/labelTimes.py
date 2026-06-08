import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from timesformer_pytorch import TimeSformer
from common.config import *
from common.redisManager import RedisManager
from .timesformer_utils import *

# 配置参数
config = {
    "model_path": DEDUCTION_MODEL_PATH+"/best_timesformer_video.pth",
    "from_csv": TRAIN_DATASETS_SAVE_PATH+"/output_small_labels.csv",
    "redis_label_name_key": "video_label_name_",
    "redis_label_time_key": "video_label_time_",
    "redis_label_confidence_key": "video_label_confidence_",
    "redis_label_mins": 10,
    "output_csv": TRAIN_DATASETS_SAVE_PATH+"/video_predictions.csv",
    "target_size": 448,
    "batch_size": 8,
    "all_region": ["all","left","center","right"] , # 从from_csv获取替换为实际类别
    "clip_length": 6,  # 时间维度帧数
    "frame_rate": 6,   # 每秒采样帧数
    "class_names": [],  # 从from_csv获取替换为实际类别
    "confidence":  NN_DEFALT_THRESH # 置信度阈值，低于此值将被忽略
}

# 区域映射定义
REGION_MAP = {
    "left": (0, 0, 0.5, 1),    # x1,y1,x2,y2 (相对坐标)
    "center": (1/4, 0, 3/4, 1),
    "right": (0.5, 0, 1, 1),
    "all": (0, 0, 1, 1)
}

# 预处理管道
transform = Compose([
    SmartResizePad(config["target_size"]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
def load_model(model_path, device):
    """加载训练好的模型"""
    # 初始化模型（需与训练时相同的架构）
    model = TimeSformer(
        dim=512,
        image_size=448,
        patch_size=16,
        num_frames=config["clip_length"],
        num_classes=len(config["class_names"]),
        depth=6,
        heads=8,
        dim_head=64
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model

def crop_by_orientation(frame, orientation):
        h, w = frame.shape[:2]
        if orientation == "left":
            return frame[:, :w//2, :]  # 左半部分
        elif orientation == "center":
            return frame[:, w//4:3*w//4, :]  # 中间1/2
        elif orientation == "right":
            return frame[:, w//2:, :]  # 右半部分
        return frame  # 默认返回完整帧

# 读取视频片段
def read_video_clip(video_path, start_time, end_time, region):
    """读取视频片段并裁剪指定区域"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # 计算绝对时间范围
    start_time = max(0, start_time)
    end_time = min(duration, end_time)
    
    # 计算区域边界
    # x1, y1, x2, y2 = REGION_MAP[region]
    # abs_x1, abs_x2 = int(x1 * width), int(x2 * width)
    # abs_y1, abs_y2 = int(y1 * height), int(y2 * height)
    
    # 计算需要采样的帧
    num_frames = int((end_time - start_time) * config["frame_rate"])
    frame_indices = np.linspace(
        start_time * fps, 
        end_time * fps, 
        num_frames, 
        dtype=int
    )
    # print(f"frame_indices----{frame_indices}")
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 裁剪区域并转换颜色空间
            # region_frame = frame[abs_y1:abs_y2, abs_x1:abs_x2]
            # region_frame = cv2.cvtColor(region_frame, cv2.COLOR_BGR2RGB)
            region_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            region_frame = crop_by_orientation(region_frame, region)
            
            frames.append(region_frame)
    
    cap.release()
    return frames

# 预处理视频片段
def preprocess_clip(frames):
    """将帧列表转换为模型输入张量"""
    transformed = []
    for frame in frames:
        # 应用空间变换
        frame = transform(frame)
        transformed.append(frame)
    
    # 堆叠帧并添加时间维度 (T, C, H, W)
    clip = torch.stack(transformed)
    return clip.unsqueeze(0)  # 添加批次维度 (1, T, C, H, W)

# 预测视频片段
def predict_video_segment(model, device, video_path, start_time, end_time,video_way:int=1,actionId:str=None,redis:RedisManager=None):
    """对视频片段在三个区域进行预测"""
    results = []
    regions=["all"]
    if video_way==1:
        regions=["all"]
    elif video_way==2:
        regions=["left","right"]
    elif video_way==3:
        regions=["left","center","right"]

    for region in regions:
        # 读取并预处理视频片段
        frames = read_video_clip(video_path, start_time, end_time, region)
        if frames is None or len(frames) < config["clip_length"]:
            # 如果帧不足，跳过该片段
            print(f"Insufficient frames for {region} region")
            continue
            
        clip = preprocess_clip(frames)
        clip = clip.to(device)
        
        # 进行推理,使用redis记录之前预测结果，考虑动作连续性
        with torch.no_grad():
            outputs = model(clip)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            
            # 获取结果
            label = config["class_names"][pred.item()]
            confidence = conf.item()
            '''
            redis_label_name=redis.get(config["redis_label_name_key"]+region)
            redis_label_time=float(redis.get(config["redis_label_time_key"]+region))
            redis_label_confidence=float(redis.get(config["redis_label_confidence_key"]+region))
            '''

            if confidence >= config["confidence"]:
                print(f"{region}区域预测结果：{label}，置信度：{confidence:.2f}")
                '''
                if redis_label_name!=label:
                    if redis_label_name!="None":
                        results.append({
                            "video_path": video_path,
                            "start_time": redis_label_time,
                            "end_time": start_time,
                            "region": region,
                            "label": redis_label_name,
                            "confidence": round(redis_label_confidence,2)
                        })
                        #actionId，动作ID回传
                        #Java端接口
                    redis.set(config["redis_label_name_key"]+region,label)
                    redis.set(config["redis_label_time_key"]+region,str(start_time))
                    redis.set(config["redis_label_confidence_key"]+region,str(confidence))
                else:
                    redis.set(config["redis_label_confidence_key"]+region,str(confidence))
                '''
                results.append({
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "region": region,
                    "label": label,
                    "confidence": confidence
                })
                #actionId，动作ID回传
                #Java端接口
            else:
                '''
                if redis_label_name!="None":
                    results.append({
                        "video_path": video_path,
                        "start_time": redis_label_time,
                        "end_time": start_time,
                        "region": region,
                        "label": redis_label_name,
                        "confidence": round(redis_label_confidence,2)
                    })
                    #Java端接口
                redis.set(config["redis_label_time_key"]+region,str(end_time))
                redis.set(config["redis_label_name_key"]+region,"None")
                redis.set(config["redis_label_confidence_key"]+region,"0.00")
                '''
                results.append({
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "region": region,
                    "label": label,
                    "confidence": confidence
                })
                #actionId，动作ID回传
                #Java端接口
                print(f"{region}区域：置信度太低: {label} ({confidence:.2f})")
    return results
'''
将视频分割为固定时长（默认1秒）的片段
可配置步长（默认1秒）控制片段重叠度,两个一样表示不重叠
'''
# 主处理函数
def process_video(video_path, actionId,window_size=1.0, step_size=1.0,video_way:int=1, redis:RedisManager=None):
    """处理视频文件并生成CSV结果"""
    # 设置设备
    device = torch.device(TIMESFORMER_DEFALT_DEVICE if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_model(config["model_path"], device)
    print(f"Loaded model on {device}")
    
    # 存储所有结果
    all_results = []
    
    # 获取视频时长
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    # 创建时间窗口
    start_times = np.arange(0, duration - window_size, step_size)
    
    # 处理每个时间窗口
    for start_time in tqdm(start_times, desc=f"Processing {os.path.basename(video_path)}", leave=False):
        end_time = start_time + window_size
        segment_results = predict_video_segment(
            model, device, video_path, start_time, end_time,video_way,actionId,redis
        )
        all_results.extend(segment_results)
        
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(all_results)
    df.to_csv(config["output_csv"], index=False)
    print(f"Results saved to {config['output_csv']}")
    if redis is not None:
        for region in config["all_region"]:
            redis.delete(config["redis_label_name_key"]+region)
            redis.delete(config["redis_label_time_key"]+region)
            redis.delete(config["redis_label_confidence_key"]+region)
    return df

def process_label(video_path,actionId,video_way:int=1, redis:RedisManager=None):
    video_path=DEDUCTION_ORG_VIDEO_PATH+"/"+video_path
    # 数据准备
    df = pd.read_csv(config["from_csv"])
    class_to_idx={cls: i for i, cls in enumerate(sorted(df.label.unique()))}
    print(f"class_to_idx-----{class_to_idx}")
    config["class_names"]= list(class_to_idx.keys())
    # print(f"class_names-----{config["class_names"]}")
    if redis is not None:
        for region in config["all_region"]:
            redis.set(config["redis_label_name_key"]+region,"None",config["redis_label_mins"]*60)
            redis.set(config["redis_label_time_key"]+region,"0.0",config["redis_label_mins"]*60)
            redis.set(config["redis_label_confidence_key"]+region,"0.00",config["redis_label_mins"]*60)
    # 处理视频并生成CSV
    results_df = process_video(video_path=video_path,actionId=actionId,video_way=video_way, redis=redis)
    
    # 打印结果摘要
    # print("\nPrediction Summary:")
    # print(results_df.groupby(["video_path", "region", "label"]).size().unstack().fillna(0))

    print(f"处理视频: {video_path}")