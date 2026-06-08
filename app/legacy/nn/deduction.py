import torch
import torch.nn as nn
import cv2
import pandas as pd
import numpy as np
from torchvision.transforms import Compose, Resize
from torchvision.io import read_video
from PIL import Image, ImageDraw, ImageFont



def cv2_put_text_chinese(img, text, position, font_size, color,font_path="simhei.ttf"):
    # 转换OpenCV图像为PIL格式（RGB）
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 加载中文字体
    font = ImageFont.truetype(font_path, font_size)
    
    # 绘制中文文本
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式（BGR）
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class ActionRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            # 输入尺寸: (3, 16, 112, 112)
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class VideoActionInference:
    def __init__(self, model_path, label_map, clip_length=6, frame_size=112):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"label_map：------{label_map}")
        self.model = self._load_model(model_path, len(label_map))
        self.label_map = label_map
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.transform = Compose([
            Resize((frame_size, frame_size)),
            lambda x: x.float() / 255.0  # 归一化
        ])
        
    def _load_model(self, model_path, num_classes):
        """加载预训练模型"""
        model = ActionRecognitionCNN(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)
    
    def _preprocess_clip(self, frames):
        """预处理视频片段"""
        # 转换为 (C, T, H, W)
        tensor_frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
        tensor_frames = self.transform(tensor_frames)
        
        # 调整时间维度
        if tensor_frames.size(1) < self.clip_length:
            # 重复填充
            repeat_factor = self.clip_length // tensor_frames.size(1) + 1
            tensor_frames = tensor_frames.repeat(1, repeat_factor, 1, 1)[:, :self.clip_length]
        else:
            # 均匀采样
            indices = torch.linspace(0, tensor_frames.size(1)-1, self.clip_length).long()
            tensor_frames = tensor_frames[:, indices]
            
        return tensor_frames.unsqueeze(0).to(self.device)  # 添加batch维度

    def process_video(self, video_path, output_csv=None, output_video=None):
        """处理完整视频"""
        # 读取视频元数据
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 结果存储
        results = []
        video_writer = None
        # 读取片段
        frames = []
        
        # 分段处理
        for start_frame in range(0, total_frames, self.clip_length):
            thisframes = []
            for i in range(self.clip_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thisframes.append(frame)
                
            
            if len(thisframes) == 0:
                break
                
            # 预处理与推理
            clip_tensor = self._preprocess_clip(np.array(thisframes))
            with torch.no_grad():
                outputs = self.model(clip_tensor)
                print(outputs)
                pred_idx = torch.argmax(outputs).item()
                # print(f"Predicted pred_idx: {pred_idx}")
                # print(f"Predicted label_map: {self.label_map}")
                label = self.label_map[pred_idx]
                
            # 计算时间戳
            start_time = start_frame / fps
            end_time = (start_frame + len(thisframes)) / fps
            
            # 保存结果
            results.append({
                "start_time": start_time,
                "end_time": end_time,
                "label": label
            })
            for i in range(self.clip_length):
                frame=thisframes[i]
                frame=cv2_put_text_chinese(frame, f"Action:{label}", (20, 40), 30, (0,255,0),"simhei.ttf")
                frames.append(frame)
            
        # 可视化处理
        if output_video:
            self._visualize_segment(frames, label, video_writer,video_path,output_video)
        
        # 输出结果
        result_df = pd.DataFrame(results)
        if output_csv:
            result_df.to_csv(output_csv, index=False,encoding="utf-8")
            
        cap.release()
        if video_writer:
            video_writer.release()
            
        return result_df
    
    def _visualize_segment(self, frames, label, video_writer,video_path, output_video):
        """生成带标注的视频"""
        cap = cv2.VideoCapture(video_path)
        h, w = frames[0].shape[:2]
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_video, 
                fourcc, 
                int(cap.get(cv2.CAP_PROP_FPS)), 
                (w, h))
                
        for frame in frames:
            # 转换颜色空间
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 添加标注
            # cv2.putText(frame, f"Action: {label}", (20, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # frame=cv2_put_text_chinese(frame, f"Action:{label}", (20, 40), 30, (0,255,0),"simhei.ttf")
            video_writer.write(frame)
    
    @classmethod
    def from_training(cls, model_path, train_csv):
        """从训练配置自动创建实例"""
        df = pd.read_csv(train_csv)
        # label_map = {i: cls for cls, i in 
        #             sorted({v:k for k,v in df.label.astype('category').cat.codes.to_dict()}.items(),
        #                    key=lambda x: x[1])}
        inverse_label_map={name: cls for cls, name in enumerate(df['label'].unique())}
        label_map = {v: k for k, v in inverse_label_map.items()}
        return cls(model_path, label_map)


# 使用示例
infer = VideoActionInference.from_training(
    model_path="action_recognition_model.pth",
    train_csv="output_small_labels.csv"
)

# 处理新视频并保存结果
result = infer.process_video(
    video_path="output_small.mp4",
    output_csv="predictions.csv",
    output_video="annotated_video.mp4"
)

print("推理完成！结果已保存至 predictions.csv")
print(result.head())