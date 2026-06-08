import os,json,time  # 原代码中使用了 os.path.join，但未导入 os
import torch,gc
import pandas as pd
import numpy as np
import torchvision  # 添加对 torchvision 的导入
from torchvision.datasets import CocoDetection
from torchvision import transforms  # 显式导入 transforms，避免潜在问题
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN,MaskRCNNPredictor, maskrcnn_resnet50_fpn,MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
import ast
import cv2
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import importlib
from typing import Optional
from torchmetrics.detection import MeanAveragePrecision



# 1. 数据预处理定义
def get_transform(train):
    transforms = []
    transforms.append(F.ToTensor())
    if train:
        # transforms.append(F.ToPILImage())
        # transforms.append(F.Resize((800, 450)))
        transforms.append(F.RandomRotation(degrees=(-30, 30)))
        transforms.append(F.RandomHorizontalFlip(0.5))
        transforms.append(F.RandomVerticalFlip(0.5))
        transforms.append(F.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ))
    return F.Compose(transforms)

# 2. 自定义数据集类
class CocoDataset(CocoDetection):
    def __init__(self, root, coco_path, transform=None, label_map=None):
        super(CocoDataset, self).__init__(root, coco_path,transform)
        self._transforms = transforms
        
        # # 创建标签映射
        # self.label_map = {name: idx+1 for idx, name in enumerate(self.df['label'].unique())}
        # # print(self.label_map)
        # self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        # 从标注文件加载类别信息
        with open(coco_path,encoding="utf-8") as f:
            coco_data = json.load(f)
        
        # 生成或使用外部传入的label_map
        if label_map is None:
            self._generate_label_map(coco_data)
        else:
            self.label_map = label_map
            self.inverse_label_map = {v: k for k, v in label_map.items()}
            self.num_classes = len(label_map)
        
        # 构建反向映射表（用于结果解析）
        self.id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.train_id_to_name = {train_id: self.id_to_name[oid] 
                                for oid, train_id in self.label_map.items()}
    def _generate_label_map(self, coco_data):
        """根据COCO数据生成类别映射表"""
        # 获取原始类别ID并排序
        original_ids = sorted([cat['id'] for cat in coco_data['categories']])
        
        # 生成从原始ID到训练ID的映射（训练ID从1开始）
        self.label_map = {oid: idx+1 for idx, oid in enumerate(original_ids)}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(original_ids) 

    def __getitem__(self, index):
        # 获取原始数据
        img, targets = super().__getitem__(index)
        
        # 转换标注格式
        formatted_targets = []
        for target in targets:
            # 转换类别ID
            original_id = target['category_id']
            train_id = self.label_map.get(original_id)
            
            if train_id is None:
                continue  # 跳过未定义类别

            formatted_targets.append({
                'boxes': torch.as_tensor(target['bbox'], dtype=torch.float32),
                'labels': torch.as_tensor(train_id, dtype=torch.int64),
                'masks': torch.as_tensor(self.coco.annToMask(target), dtype=torch.uint8),
                'image_id': torch.tensor([target['image_id']]),
                'area': torch.as_tensor(target['area']),
                'iscrowd': torch.as_tensor(target['iscrowd'])
            })

        # 应用数据增强
        if self.transforms is not None:
            img, formatted_targets = self.transforms(img, formatted_targets)

        return img, formatted_targets

    @property
    def class_names(self):
        """获取有序类别名称列表"""
        return [self.train_id_to_name[i+1] for i in range(self.num_classes)]


 # 自定义collate_fn处理空样本
def collate_fn(batch):
    return tuple(zip(*[item for item in batch if item is not None]))

# 实时跟踪训练指标（损失、时间等）
class MetricLogger:
    """用于记录和统计训练指标的类"""
    def __init__(self):
        self.meters = {}
        self.start_time = time.time()
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def log_every(self, iterable, print_freq):
        """定期打印日志的迭代器包装器"""
        iter_time = AverageMeter()
        data_time = AverageMeter()
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        
        start_time = time.time()
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - start_time)
            yield obj
            iter_time.update(time.time() - start_time)
            start_time = time.time()

            if (i + 1) % print_freq == 0:
                log_str = self._get_log_str(i, len(iterable), iter_time, data_time)
                print(log_str)
                self.reset()

    def _get_log_str(self, iter, total_iter, iter_time, data_time):
        eta_seconds = iter_time.global_avg * (total_iter - iter - 1)
        eta_string = str(time.strftime("%H:%M:%S", time.gmtime(eta_seconds)))
        
        log_str = (
            f"Iter: [{iter+1}/{total_iter}]  "
            f"Time: {iter_time.avg:.3f} ({iter_time.global_avg:.3f})  "
            f"Data: {data_time.avg:.3f} ({data_time.global_avg:.3f})  "
            f"ETA: {eta_string}"
        )
        for name, meter in self.meters.items():
            log_str += f"  {name}: {meter.avg:.4f}"
        return log_str

    @property
    def loss(self):
        return self.meters.get('loss', AverageMeter())

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

class AverageMeter:
    """计算并存储平均和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def global_avg(self):
        return self.sum / self.count if self.count != 0 else 0

def get_model_instance_segmentation(label_map,num_classes):
    
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,num_classes=num_classes)
    # 修改分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(
        in_features, num_classes
    )
    # 修改掩码头
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,  # hidden_layer
        num_classes
    )
    model.label_map = label_map
    return model
# 4. 训练函数
def train_model(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 创建数据集
    dataset_train = CocoDataset(
        root=config['img_dir'],
        annFile=config['train_ann_file'],
        transform=get_transform(train=True))
    
    dataset_val = CocoDataset(
        root=config['img_dir'],
        annFile=config['val_ann_file'],
        transform=get_transform(train=False))
    
    # 创建数据加载器
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn)
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn)
    
    # 初始化模型
    num_classes = config['num_classes'] + 1  # +1 for background
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # 优化器和学习率调度器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['lr'],
        momentum=0.9,
        weight_decay=0.0005)
    
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1)
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        metric_logger = MetricLogger()
        
        # 训练阶段
        for images, targets in data_loader_train:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            metric_logger.update(loss=losses, **loss_dict)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in data_loader_val:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        
        # 更新学习率
        lr_scheduler.step()
        
        # 保存模型
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save(model.state_dict(), 
                      f"{config['model_dir']}/model_epoch{epoch+1}.pth")
        print(f"Epoch {epoch+1} | Train Loss: {metric_logger.loss.global_avg:.4f} | Val Loss: {val_loss/len(data_loader_val):.4f}")




####
def hahaha():
    print("hahaha")