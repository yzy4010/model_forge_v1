import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from timesformer_pytorch import TimeSformer
from common.redisManager import RedisManager
from .timesformer_utils import *
from torchvision.transforms import *
from common.config import *

# 配置参数
config = {
    "csv_path": TRAIN_DATASETS_SAVE_PATH+"/output_small_labels.csv",
    # "train_type": "1", #训练推理模式：1全画面训练，2左右分开，3左中右分开
    # "frame_size": [640,360],  # 16*9，当前方案不用
    "target_size":224,# 短边缩放大小，一般是224，这里为了更精确识别，设为640
    "frame_rate": 15,   # 每秒截取帧数
    "batch_size": 16,
    "num_workers": 16,
    "num_epochs": 20,
    "num_classes": 3,
    "val_ratio": 0.2,
    "eval_interval": 1,
    "lr":0.0004,
    "weight_decay":0.05,
    "model_config": {
        "dim": 512,
        "image_size": 224,
        "patch_size": 16,
        "num_frames": 15,
        "num_classes": 3,
        "depth": 6,
        "heads": 8,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1
    }
}



# 自定义数据集类（适配TimeSformer）
class VideoActionDataset(Dataset):
    def __init__(self, df,is_train=True):
        self.df = df.reset_index(drop=True)
        # print("class_to_idx---->",self.class_to_idx)
        self.is_train = is_train
        if is_train:
            config["num_classes"]=len(self.class_to_idx)
            config["model_config"]["num_classes"]=len(self.class_to_idx)

        self.transform = self._default_transform()

    def _default_transform(self):
        transforms = [SmartResizePad(target_size=config["target_size"])]  # 核心修改点
        
        if self.is_train:
            transforms += [
                ToPILWrapper(),
                RandomHorizontalFlip(p=0.5),
                # 可添加其他空间增强但需注意不裁剪
                ColorJitter(brightness=0.2, contrast=0.2,saturation=0.2)
            ]
            
        transforms += [
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
        ]
        return Compose(transforms)

    def __len__(self):
        return len(self.df)
    
    @property
    def class_to_idx(self):
        return {cls: i for i, cls in enumerate(sorted(self.df.label.unique()))}
    
    def _crop_by_orientation(self,frame, orientation):
        h, w = frame.shape[:2]
        if orientation == "left":
            return frame[:, :w//2, :]  # 左半部分
        elif orientation == "center":
            return frame[:, w//4:3*w//4, :]  # 中间1/2
        elif orientation == "right":
            return frame[:, w//2:, :]  # 右半部分
        return frame  # 默认返回完整帧

    def _sample_frames(self, cap, start_frame, end_frame,orientation):
        frame_indices = np.linspace(start_frame, end_frame, config["frame_rate"], dtype=int)
        # print(f"frame_indices----{frame_indices}")
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self._crop_by_orientation(frame, orientation)
                frame = self.transform(frame)
                frames.append(frame)
        return torch.stack(frames)  # (T, C, H, W)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cap = cv2.VideoCapture(DEDUCTION_ORG_VIDEO_PATH+"/"+row.video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int((row.start_time * fps)+(fps/config["frame_rate"]))
        end_frame = int(row.end_time * fps)
        orientation=row.orientation
        # num_frames = int((row.end_time - row.start_time) * config["frame_rate"])
        
        clip = self._sample_frames(cap, start_frame, end_frame,orientation)
        cap.release()
        
        # label_map = {"jump":0, "run":1, "kick":2}
        label = row.label
        label_idx = self.class_to_idx[label]

        return clip, torch.tensor(label_idx)

# 评估函数（支持mAP和准确率）
def evaluate(model, dataloader,device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            outputs = model(clips)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算mAP
    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    
    ap_scores = []
    for class_id in range(config["num_classes"]):
        ap = average_precision_score(
            (y_true == class_id).astype(int),
            y_score[:, class_id]
        )
        ap_scores.append(ap)
    mean_ap = np.mean(ap_scores)
    
    # 计算准确率
    accuracy = (y_true == np.argmax(y_score, axis=1)).mean()
    
    return {
        "mAP": mean_ap,
        "accuracy": accuracy,
        "probabilities": y_score,
        "labels": y_true
    }

# 训练流程（修改了优化策略）
def train_from_scratch(redis:RedisManager):
    # 数据准备
    df = pd.read_csv(config["csv_path"])
    # train_df, val_df = train_test_split(df, test_size=config["val_ratio"], random_state=42)
    # intAry=np.linspace(305, 600, 60, dtype=int)
    # print(f"intAry-----{intAry}")

    # 创建数据加载器
    train_dataset = VideoActionDataset(df,True)
    val_dataset = VideoActionDataset(df,False)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"],shuffle=False)

    # 初始化TimeSformer模型
    device = torch.device(TIMESFORMER_DEFALT_DEVICE if torch.cuda.is_available() else "cpu")
    model = TimeSformer(
        dim=config["model_config"]["dim"],
        image_size=config["model_config"]["image_size"],
        patch_size=config["model_config"]["patch_size"],
        num_frames=config["model_config"]["num_frames"],
        num_classes=config["model_config"]["num_classes"],
        depth=config["model_config"]["depth"],
        heads=config["model_config"]["heads"],
        dim_head=config["model_config"]["dim_head"],
        attn_dropout=config["model_config"]["attn_dropout"],
        ff_dropout=config["model_config"]["ff_dropout"]
    ).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # 训练循环
    best_map = 0.0
    for epoch in range(config["num_epochs"]):
        # print(f"Epoch {epoch+1}/{config['num_epochs']}")
        redis.set(START_TRAINING_PX + "video_epoch", str(epoch), 3600)
        model.train()
        running_loss = 0.0
        
        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        # 验证阶段
        if (epoch+1) % config["eval_interval"] == 0:
            val_metrics = evaluate(model, val_loader,device)
            train_loss = running_loss / len(train_loader)
            
            print(f"Epoch {epoch+1}/{config['num_epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val mAP: {val_metrics['mAP']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # 保存最佳模型
            if val_metrics["mAP"] > best_map:
                best_map = val_metrics["mAP"]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': best_map,
                }, DEDUCTION_MODEL_PATH+"/best_timesformer_video.pth")
        
        if redis and redis.get(f"{START_TRAINING_PX}video")=="end":
            print("训练任务终止结束了")
            break

    # 最终测试
    final_metrics = evaluate(model, val_loader,device)
    print(f"\nFinal Performance:")
    print(f"mAP: {final_metrics['mAP']:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    if redis:
        redis.delete(START_TRAINING_PX+"video")

    # 可视化混淆矩阵
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # cm = confusion_matrix(final_metrics["labels"], np.argmax(final_metrics["probabilities"], axis=1))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["再检查", "插卡", "放盖", "放笔", "检查", "退检", "锤紧"])
    # disp.plot(cmap=plt.cm.Blues)
    # plt.savefig("confusion_matrix.png")
