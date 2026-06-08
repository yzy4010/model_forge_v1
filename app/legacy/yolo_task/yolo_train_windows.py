import sys
import os

# 添加项目根目录到Python路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)
import argparse
import os, platform
import time
import threading
from typing import Optional
import pandas as pd
import json
import csv
from fastapi import HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from common.ajaxResult import postAsyncJsonWithOutJwt, postJsonWithOutJwt
from common.redisManager import RedisManager
from model.videoInput import TrainParameter
from common.config import *
from ultralytics.engine.trainer import BaseTrainer
import torch
from common.runScript import *
import yaml
import shutil
from PIL import Image

# 判断运行程序是 Linux还是Windows
osName = platform.system().lower()


# 定义与TrainParameter匹配的Pydantic模型
class ModelConfig(BaseModel):
    epochs: int = 5
    imgSiz: int = 640
    batch: int = 4
    lr: float = 0.01
    momentum: float = 0.9
    weightDecay: float = 0.0004


class TrainRecordInfo(BaseModel):
    trainingRecordListId: str = ""
    trainingListId: str = ""
    tranName: str = ""
    actionId: str = ""
    actionName: str = ""


class TrainParameter(BaseModel):
    modelType: Optional[str] = None
    modelConfig: ModelConfig
    trainRecords: list[TrainRecordInfo]


# coco数据集转csv文件
# coco_json_path  原始coco数据集目录
# output_csv_path 输出目录
def coco_to_csv(coco_json_path, output_csv_path):
    # 加载COCO数据
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 创建映射字典
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # 准备写入CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 合并表头
        writer.writerow(
            ['filename', 'timestamp', 'label',
             'xmin', 'ymin', 'xmax', 'ymax',  # 边界框坐标
             'segmentation'  # 分割多边形
             ]
        )

        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            img_info = image_id_to_info.get(image_id)
            if not img_info:
                continue

            # ========== 通用处理 ==========
            # 提取帧数和时间戳
            filename = img_info['file_name']
            # frame_match = re.search(r'frame_(\d+)\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
            # if not frame_match:
            #     print(f"跳过无法解析帧数的文件: {filename}")
            #     continue
            # frame_number = int(frame_match.group(1))
            # timestamp = round(frame_number / 6.0, 2)  # 6帧/秒
            # 获取类别
            label = category_id_to_name.get(ann['category_id'], 'unknown')

            # ========== BBOX处理 ==========
            x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
            xmin = round(x, 1)  # 保留1位小数
            ymin = round(y, 1)
            xmax = round(x + w, 1)
            ymax = round(y + h, 1)

            # 处理segmentation（取第一个多边形）
            seg_str = ''
            if ann['segmentation']:
                polygon = ann['segmentation'][0]
                # print(type(polygon))
                if not isinstance(polygon, list):
                    polygon = ann['segmentation']
                points = [f"{round(polygon[i], 1)},{round(polygon[i + 1], 1)}" for i in range(0, len(polygon), 2)]
                seg_str = ';'.join(points)

            # ========== 写入结果 ==========
            writer.writerow([filename,
                             0, label,
                             xmin, ymin, xmax, ymax,
                             seg_str
                             ])

    print("转换完成！")


# csv转换训练和验证
def csv_to_yolo(images_dir, csv_path, output_dir, img_size=None):
    """
    参数说明：
    csv_path: 输入的CSV文件路径
    output_dir: 输出目录（自动创建images/train, labels/train等子目录）
    images_dir: 图片地址
    img_size: 兼容参数，默认按每张图真实尺寸归一化
    """

    # 创建目录结构
    dirs = {
        'images': ['train', 'val'],
        'labels': ['train', 'val']
    }
    for main_dir, sub_dirs in dirs.items():
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(output_dir, main_dir, sub_dir), exist_ok=True)

    # 读取CSV数据
    df = pd.read_csv(csv_path)

    # 获取所有唯一文件名
    unique_files = df['filename'].unique()

    # 划分训练集和验证集（8:2比例）
    train_files, val_files = train_test_split(unique_files, test_size=0.2, random_state=42)

    # 创建类别映射
    classes = df['label'].unique().tolist()
    class_map = {name: idx for idx, name in enumerate(classes)}

    # 处理每个文件
    for file_group, split_name in [(train_files, 'train'), (val_files, 'val')]:
        for filename in file_group:
            # 复制图像文件
            # src_img_path = os.path.join('frames', filename)  # 假设原始图像在frames目录
            src_img_path = os.path.join(images_dir, filename)  # 假设原始图像在frames目录
            dst_img_dir = os.path.join(output_dir, 'images', split_name)
            # 先校验图片是否存在
            if not os.path.exists(src_img_path):
                print(f"警告：图片不存在，跳过 {src_img_path}")
                continue
            # 再读取图片尺寸
            with Image.open(src_img_path) as img:
                img_w, img_h = img.size
            if img_w <= 0 or img_h <= 0:
                print(f"警告：图片尺寸非法，跳过 {src_img_path}")
                continue
            # 校验通过后复制图片
            shutil.copy(src_img_path, dst_img_dir)

            # 创建标注文件
            txt_path = os.path.join(output_dir, 'labels', split_name,
                                    os.path.splitext(filename)[0] + '.txt')

            # 提取该图像的所有标注
            img_data = df[df['filename'] == filename]
            with open(txt_path, 'w', encoding="utf-8") as f:
                for _, row in img_data.iterrows():
                    # 转换多边形坐标
                    points = [list(map(float, p.split(',')))
                              for p in row['segmentation'].split(';')]

                    # 归一化坐标（相对于图像尺寸）
                    normalized = []
                    for x, y in points:
                        nx = x / img_w
                        ny = y / img_h
                        nx = max(0.0, min(1.0, nx))
                        ny = max(0.0, min(1.0, ny))
                        normalized.extend([nx, ny])

                    # region  class_mapping 弃用
                    # 获取类别ID（需要提前定义类别映射）对应 class_mapping 弃用
                    # class_id = class_mapping[row['label']]
                    # # 写入YOLO格式：class_id x1 y1 x2 y2 ...
                    # line = f"{class_id} " + " ".join(map(str, normalized)) + "\n"
                    # f.write(line)
                    # endregion

                    # 获取类别ID 写入标注
                    class_id = class_map[row['label']]
                    f.write(f"{class_id} " + " ".join(map(str, normalized)) + "\n")
    # 保存类别信息 必须要这个替换原本的 data.yml 配置  不然跑不起来
    with open(os.path.join(output_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"names: {classes}\n")
        f.write(f"nc: {len(classes)}\n")
        # f.write(f"path: {os.path.abspath(os.path.join(output_dir))}\n")
        f.write(f"train: {os.path.abspath(os.path.join(output_dir, 'images/train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(output_dir, 'images/val'))}\n")


# 自定义日志回调 并发送java接口回传训练日志
class AddCirclePassTrainLog:
    def __init__(self, trainingRecordId, redis: RedisManager):
        self.trainingRecordId = trainingRecordId
        self.redis = redis

    def on_train_epoch_end(self, trainer: BaseTrainer):
        # 获取当前 epoch 和指标
        epoch = trainer.epoch + 1
        map50 = trainer.validator.metrics.mean_results()[0]  # 获取 mAP50-95
        metrics = trainer.metrics
        # box_loss = metrics.get('train/box_loss', 0)
        # cls_loss = metrics.get('train/cls_loss', 0)
        # dfl_loss = metrics.get('train/dfl_loss', 0)
        # val_box_loss = metrics.get('val/box_loss', 0)
        # val_cls_loss = metrics.get('val/cls_loss', 0)
        # 获取基础训练指标
        # • box_loss：边界框回归损失
        # • cls_loss：分类损失
        # • dfl_loss：分布焦点损失（某些版本特有）
        # GPU显存使用量(GB)
        gpu_mem = torch.cuda.memory_allocated() / 1E9  # 转换为GB单位
        # 训练边界框回归损失
        box_loss = trainer.loss_items[0]
        # 训练分类损失
        cls_loss = trainer.loss_items[1]
        # 分布焦点损失
        dfl_loss = trainer.loss_items[2] if len(trainer.loss_items) > 2 else 0
        # 验证边界框回归损失
        val_box_loss = metrics.get('val/box_loss', 0)
        # 验证分类损失
        val_cls_loss = metrics.get('val/cls_loss', 0)
        # 获取模型训练阶段整体损失值
        val_loss = trainer.loss
        # 获取批次大小
        batch_size = trainer.batch_size
        # 输入图像尺寸
        img_size = trainer.args.imgsz if hasattr(trainer.args, 'imgsz') else 640
        # 获取学习率
        lr = trainer.optimizer.param_groups[0]['lr']
        # 获取
        print(f'epoch:{epoch}')
        print(f'GPU_mem:{gpu_mem:.4f}')
        print(f'box_loss:{box_loss:.4f}')
        print(f'cls_loss:{cls_loss:.4f}')
        print(f'dfl_loss:{dfl_loss:.4f}')
        print(f'val_box_loss:{val_box_loss:.4f}')
        print(f'val_cls_loss:{val_cls_loss:.4f}')
        print(f'val_loss:{val_loss:.4f}')
        print(f'lr:{lr:.6f}')
        print(f'batch_size :{batch_size}')
        print(f'img_size:{img_size}')
        print(f"current_epoch: {epoch}")
        print(f"map50: {float(map50)}")
        print(f"trainingRecordId: {self.trainingRecordId}")
        if map50 is None or map50 == 0:
            map50 = 0
        # 调用java接口回传训练日志
        threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_TRAIN_LOG_TO_JAVA, {
            "trainingRecordListId": self.trainingRecordId,
            "mAP": f"{map50:.6f}",
            "epoch": epoch,
            "logLv": "1",
            "logsContent": f"第{epoch}轮训练记录,训练质量{map50:.6f},GPU显存使用量{gpu_mem:.6f},训练边界框回归损失{box_loss:.6f},"
                           f"训练分类损失{cls_loss:.6f},分布焦点损失{dfl_loss:.6f},验证边界框回归损失{val_box_loss:.6f},"
                           f"验证分类损失{val_cls_loss:.6f},整体损失值{val_loss:.6f},训练批次大小{batch_size},训练图像尺寸{img_size},学习率{lr:.6f}",
        })).start()
        if self.redis and self.redis.get(f"{START_TRAINING_PX}{self.trainingRecordId}") == "end":
            print("训练任务终止结束了")
            print(self.redis.get(f"{START_TRAINING_PX}{self.trainingRecordId}"))
            trainer.stop = True  # 停止训练


# 删除指定目录下得文件
def delete_files(directory, pattern):
    """
    删除指定目录中匹配pattern的所有文件
    :param directory: 目标目录路径
    :param pattern: 文件名匹配模式（支持通配符*.xxx）
    """
    for filename in os.listdir(directory):
        if filename.endswith(pattern) or filename == pattern:  # 支持后缀匹配或全名匹配
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")


def reset_cache(dataset_path):
    cache_files = [
        os.path.join(dataset_path, "labels/train.cache"),
        os.path.join(dataset_path, "labels/val.cache")
    ]
    for f in cache_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed: {f}")

    # 创建空目录触发重建
    os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=True)
    print("Cache reset complete")


# 训练方法
def action_first_argparse(request: TrainParameter, trainingRecord_id: str, action_name: str,
                          action_id: str, redis: RedisManager):
    print("coco数据集转csv文件")

    coco_json_path = TRAIN_DATASETS_SAVE_PATH + '/' + action_id + '.json'
    output_csv_path = TRAIN_DATASETS_SAVE_PATH + '/' + action_id + '.csv'
    coco_to_csv(coco_json_path, output_csv_path)

    print("解析csv文件")
    images_dir = TRAIN_IMAGES_SAVE_PATH
    csv_path = TRAIN_DATASETS_SAVE_PATH + '/' + action_id + '.csv'
    output_dir = TRAIN_YOLO_TEMPDATA_PATH + '/' + trainingRecord_id
    # output_dir = TRAIN_YOLO_TEMPDATA_PATH + '/' + action_id
    csv_to_yolo(images_dir, csv_path, output_dir)
    # 验证配置文件是否存在
    if not os.path.exists(TRAIN_YOLO_TEMPDATA_PATH):
        raise HTTPException(
            status_code=400,
            detail=f"Data config file {output_dir + '/data.yaml'} not found."
        )

    print("加载训练模型,开始训练")
    train_model = DEDUCTION_MODEL_PATH + '/' + action_name + ".pt"

    if os.path.isfile(train_model):
        '''如果存在加载模型然后读取yaml文件判断生成得yaml文件内容和当前模型得分类id以及分类名称是否一致'''
        model = YOLO(train_model)
        # 获取模型分类id和分类名称信息
        model_nc = model.model.nc if hasattr(model, 'model') else model.nc
        model_classes = model.names
        # 读取YAML文件
        yaml_path = output_dir + '/data.yaml'
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # 获取YAML文件分类id和分类名称信息
        yaml_nc = len(data['names'])
        yaml_classes = data['names']
        # 验证类别名称是否一致
        mismatch_found = False
        for class_id in range(model_nc):
            model_name = model_classes.get(class_id, "未定义")
            yaml_name = yaml_classes[class_id] if class_id < len(yaml_classes) else "未定义"
            if model_name != yaml_name:
                print(f"⚠️ ID {class_id} 不匹配: 模型='{model_name}' ≠ YAML='{yaml_name}'")
                mismatch_found = True
        # 判断如果分类名称不一致或者模型分类id不一致就删除这个模型并加载初始模型否则继续训练
        if mismatch_found or model_nc != yaml_nc:
            print("模型分类名称或者模型分类id不一致")
            # delete_files(DEDUCTION_MODEL_PATH, action_name + ".pt")
            model = YOLO(TRAIN_ORGMODEL_PATH + '/yolov8n-seg.pt')
            print(f"加载初始模型")
    else:
        print(f"文件 {train_model} 不存在")
        model = YOLO(TRAIN_ORGMODEL_PATH + '/yolov8n-seg.pt')

    print(f"训练参数: {request.modelConfig}")

    # 创建回调实例
    java_callback = AddCirclePassTrainLog(
        trainingRecordId=trainingRecord_id,
        redis=redis
    )
    # 注册回调到训练器
    model.add_callback("on_train_epoch_end", java_callback.on_train_epoch_end)

    results = model.train(
        data=output_dir + '/data.yaml',
        epochs=request.modelConfig.epochs,
        imgsz=request.modelConfig.imgSiz,
        batch=request.modelConfig.batch,
        project=TRAIN_MODEL_SAVE_PATH,
        name=trainingRecord_id,
        device=YOLOV8_DEVICE,
        verbose=True,
        lr0=request.modelConfig.lr,  # 初始学习率
        lrf=0.01,  # 最终学习率衰减系数
    )

    # # 从结果中提取指标
    final_path = results.save_dir
    final_map50 = results.results_dict["metrics/mAP50(B)"]
    final_map = results.results_dict["metrics/mAP50-95(B)"]
    print(f"文件保存路径: {final_path}")
    print(f"Final mAP50: {final_map50}, mAP50-95: {final_map}")
    if redis and redis.get(f"{START_TRAINING_PX}{trainingRecord_id}") == "end":
        print("训练任务终止结束了")
        print(redis.get(f"{START_TRAINING_PX}{trainingRecord_id}"))
        # 调用java接口回传训练结果
        threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_TRAIN_RESULT_TO_JAVA, {
            "trainingRecordListId": trainingRecord_id,
            "mAP": f'{final_map:.6f}',
            "circle": 0
        })).start()
    else:
        print("正常停止")
        # 调用java接口回传训练结果
        threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_TRAIN_RESULT_TO_JAVA, {
            "trainingRecordListId": trainingRecord_id,
            "mAP": f'{final_map:.6f}',
            "circle": 1
        })).start()
    # 本地源文件
    localSavePath = TRAIN_MODEL_SAVE_PATH + "/" + trainingRecord_id + "/weights/last.pt"
    # 推送后的文件
    remote_path = TRAIN_MODEL_SAVE_PATH + "/" + trainingRecord_id + ".pt"

    if osName == "linux":
        # or osName=="darwin"  苹果   windows
        # 将模型训练文件推送到推理服务器
        time.sleep(1)
        scp_push(local_path=localSavePath,
                 remote_path=remote_path,
                 remoteServer=linuxServer(
                     isNeedSudo=True,
                     ip=DEDUCTION_SERVER_IP,
                     port=DEDUCTION_SERVER_SSH_PORT,
                     user=DEDUCTION_SERVER_SSH_USER,
                     password=DEDUCTION_SERVER_SSH_PASSWORD
                 ),
                 local_password=TRAIN_SERVER_SSH_PASSWORD,
                 shell=True)


def action_first_te():
    model = YOLO('D:/data/train/model/actionFirst/yolov8n-seg.pt')
    results = model.train(
        name="ce",
        project="/data/train/model/actionList",
        data="/data/train/tempData/yoloData/edcf584cf48c48e7a065a5f296420ce6/data.yaml",
        batch=4,  # 根据 GPU 显存调整
        epochs=220,
        device="cuda",
        imgsz=640,
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率衰减系数
    )


def main():
    parser = argparse.ArgumentParser(description="YOLOv8训练脚本")

    # 添加所有必需参数
    parser.add_argument("--model_type", type=str, default="1", help="模型类型")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--img_siz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--batch", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="动量参数")
    parser.add_argument("--weight_decay", type=float, default=0.0004, help="权重衰减")

    # TrainRecordInfo参数
    parser.add_argument("--training_record_id", required=True, help="训练记录ID")
    parser.add_argument("--train_id", required=True, help="训练ID")
    parser.add_argument("--tran_name", required=True, help="训练名称")
    parser.add_argument("--action_id", required=True, help="记录动作ID")
    parser.add_argument("--action_name", required=True, help="记录动作名称")

    # python yolo_train_test_1.py  --training_record_id "1"   --action_id "edcf584cf48c48e7a065a5f296420ce6" --epochs 10 --img_size 640 --batch 8 --lr 0.001
    args = parser.parse_args()
    # 创建 Redis 实例
    redis = RedisManager()
    redis_key = START_TRAINING_PX + args.training_record_id
    redis.set(redis_key, "begin", 3600)

    # 构建复杂对象
    model_config = ModelConfig(
        epochs=args.epochs,
        imgSiz=args.img_siz,
        batch=args.batch,
        lr=args.lr,
        momentum=args.momentum,
        weightDecay=args.weight_decay
    )

    train_record = TrainRecordInfo(
        trainingRecordListId=args.training_record_id,
        trainingListId=args.train_id,
        tranName=args.tran_name,
        actionId=args.action_id,
        actionName=args.action_name
    )

    request = TrainParameter(
        modelType=args.model_type,
        modelConfig=model_config,
        trainRecords=[train_record]  # 包装成列表
    )
    # 调用目标函数
    action_first_argparse(
        request=request,
        trainingRecord_id=args.training_record_id,
        action_name=args.action_id + '_' + args.model_type,
        action_id=args.action_id,
        redis=redis
    )


# python '.\yolo_task\yolo_train_test_1.py' --model_type 2 --epochs 5 --img_siz 640 --batch 4 --lr 0.01  --training_record_id 'df1051ac222a4bd685406cc9e1f9fbea' --train_id '1bec3ac42c4d4ff6a2ae3d538cb465ab' --tran_name '第三工艺yolo' --rec_action_id 'edcf584cf48c48e7a065a5f296420ce6' --rec_action_name '三工艺'


if __name__ == "__main__":
    main()
