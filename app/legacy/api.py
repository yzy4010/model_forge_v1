"""旧版 API 端点。

从 python/main.py 提取，保持原有路径和响应格式不变。
响应格式：{"code": 200, "msg": "success", "data": ...}
"""

import subprocess
from collections import deque
import threading
from sys import platform
import uuid
import time
import os
import json
from datetime import datetime, timedelta
from threading import Event as ThreadPoolEvent
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 旧版工具类和模型
from app.legacy.common.ajaxResult import ajaxReturnYes, ajaxReturnNo
from app.legacy.common.threadTaskManager import ThreadTaskManager
from app.legacy.common.config import *
from app.legacy.common.process import start_empty_video
from app.legacy.common.redisManager import RedisManager

# 旧版数据模型
from app.legacy.model.videoInput import *

# RCNN 模块（通过 sys.path 导入）
try:
    from rcn.labelAction import labelAction
    from rcn.trainAction import startTrain, create_model
except ImportError:
    labelAction = None
    startTrain = None
    create_model = None

# NN 模块
try:
    from nn.initTrain import *
    from nn.labelTimes import *
except ImportError:
    pass

# YOLO 任务
try:
    from yolo_task.yolo_process import start_process_video
except ImportError:
    start_process_video = None

# 旧版视频流处理
try:
    from app.legacy.process_videoStream import SaveMixVideos
except ImportError:
    SaveMixVideos = None

legacy_router = APIRouter(tags=["旧版接口"])

task_manager = ThreadTaskManager(max_workers=20, thread_name_prefix="main")

# 模块级 Redis 实例（旧版各端点共用）
_legacy_redis = RedisManager()


def get_os_type():
    """判断当前操作系统类型"""
    if platform.startswith('win'):
        return "Windows"
    elif platform.startswith('linux'):
        return "Linux"
    else:
        return "Other"


def run_auto_label_maskrcnn(stop_event: ThreadPoolEvent, LabelConfig: labelConfig, redis: RedisManager):
    """启动预标记（Mask R-CNN）"""
    try:
        if stop_event.is_set():
            print("预标记启动失败")
            return
        if labelAction:
            labelAction(LabelConfig, redis)
    except Exception as e:
        raise RuntimeError(f"预标记启动失败: {str(e)}")


def run_auto_label_torchnn(stop_event: ThreadPoolEvent, videoPath: str, actionId: str, redis: RedisManager):
    """启动视频预标记（Torch NN）"""
    try:
        if stop_event.is_set():
            print("预标记启动失败")
            return
        process_label(videoPath, actionId, 2, redis)
    except Exception as e:
        raise RuntimeError(f"视频预标记启动失败: {str(e)}")


def run_auto_label_yolo(stop_event: ThreadPoolEvent, LabelConfig: labelConfig, redis: RedisManager):
    """启动预标记（YOLO）"""
    try:
        if stop_event.is_set():
            print("预标记启动失败")
            return
        if start_process_video:
            start_process_video(LabelConfig, redis)
    except Exception as e:
        raise RuntimeError(f"预标记启动失败: {str(e)}")


def run_auto_label_empty(stop_event: ThreadPoolEvent, LabelConfig: labelConfig, redis: RedisManager):
    """启动预标记（空白）"""
    try:
        if stop_event.is_set():
            print("预标记启动失败")
            return
        start_empty_video(LabelConfig, redis)
    except Exception as e:
        raise RuntimeError(f"预标记启动失败: {str(e)}")


# region 视频预标记
@legacy_router.post("/startVideoPreMarking")
def start_video_pre_marking(videoActionInfo: videoActionInfo):
    redis = _legacy_redis
    LabelConfig = labelConfig("", "1", videoActionInfo.videoFile, videoActionInfo.videoType, videoActionInfo.guId, False)
    if len(videoActionInfo.actions) == 0:
        redis_id = START_VIDEO_PRE_MARK_PX + "0000"
        redis.set(redis_id, "begin", 3600)
        task_id = LabelConfig.guid + "_0000"
        task_manager.start_task(run_auto_label_empty, 0, task_id, LabelConfig, redis)
        return ajaxReturnYes(None)
    else:
        ii = 0
        for action in videoActionInfo.actions:
            ii = ii + 1
            if ii == 1:
                LabelConfig.isGetNoLabel = True
            LabelConfig.id = action.id
            LabelConfig.modelType = action.modelType
            redis_id = START_VIDEO_PRE_MARK_PX + LabelConfig.id
            redis.set(redis_id, "begin", 3600)
            task_id = LabelConfig.guid + "_" + LabelConfig.id
            if action.modelType == "1":
                task_manager.start_task(run_auto_label_maskrcnn, 0, task_id, LabelConfig, redis)
            elif action.modelType == "2":
                task_manager.start_task(run_auto_label_yolo, 0, task_id, LabelConfig, redis)
    return ajaxReturnYes(None)


@legacy_router.post("/stopVideoPreMarking")
def stop_video_pre_marking(videoActionInfo: videoActionInfo):
    redis = _legacy_redis
    redis.set(START_VIDEO_PRE_MARK_PX + "0000", "end", 30)
    for action in videoActionInfo.actions:
        redis_id = START_VIDEO_PRE_MARK_PX + action.id
        redis.set(redis_id, "end", 30)
        task_id = videoActionInfo.guId + "_" + action.id
        task_manager.stop_task(task_id)
        task_manager.stop_task(task_id + "_video")
    return ajaxReturnYes(None)
# endregion


# region 训练停止
@legacy_router.post("/stopTraining")
def stop_training(stopTrainInput: StopTrainInput):
    redis = _legacy_redis
    print("停止训练")
    task_id = stopTrainInput.trainingRecordListId
    redis_key = START_TRAINING_PX + task_id
    task_manager.stop_task(task_id)
    redis.set(redis_key, "end", 3600)
    return ajaxReturnYes(None)
# endregion


# region YOLO 训练
def train_task_yolo(stop_event: ThreadPoolEvent, train: TrainParameter, csv_name: str, data_id: str,
                    project_name: str, action_name: str, action_id: str, redis: RedisManager):
    try:
        osName = get_os_type()
        print("当前操作系统:", osName)
        if osName == "Windows":
            for obj in train.trainRecords:
                command = [
                    "python",
                    os.path.join(os.path.dirname(__file__), "yolo_task", "yolo_train_windows.py"),
                    "--model_type", f"{train.modelType}",
                    "--epochs", f"{train.modelConfig.epochs}",
                    "--img_siz", f"{train.modelConfig.imgSiz}",
                    "--batch", f"{train.modelConfig.batch}",
                    "--lr", f"{train.modelConfig.lr}",
                    "--training_record_id", f"{obj.trainingRecordListId}",
                    "--train_id", f"{obj.trainingListId}",
                    "--tran_name", f"{obj.tranName}",
                    "--action_id", f"{obj.actionId}",
                    "--action_name", f"{obj.actionName}"
                ]
                print("命令参数：", command)
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True, errors='replace')
                    print("执行成功！")
                    print("输出内容：", result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"执行失败！错误代码：{e.returncode}")
                    print("错误信息：", e.stderr)
                except FileNotFoundError:
                    print("错误：找不到python解释器或脚本文件")
                print("开始训练")
        else:
            try:
                from yolo_task.yolo_train import action_first
                action_first(train, csv_name, data_id, project_name, action_name, action_id, redis)
            except ImportError:
                print("Linux YOLO 训练脚本不可用")
        if stop_event.is_set():
            print("检测到停止请求，停止训练")
            return
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise RuntimeError(f"处理失败: {str(e)}")
# endregion


# region Mask R-CNN 训练
def train_task_maskrcnn(stop_event: ThreadPoolEvent, trainConfig, redis: RedisManager):
    try:
        if startTrain:
            startTrain(trainConfig, redis)
        if stop_event.is_set():
            print("检测到停止请求，停止训练")
            return
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise RuntimeError(f"处理失败: {str(e)}")
# endregion


# region Torch-NN 训练
def train_task_nn(stop_event: ThreadPoolEvent, redis: RedisManager):
    try:
        train_from_scratch(redis)
        if stop_event.is_set():
            print("检测到停止请求，停止训练")
            return
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise RuntimeError(f"处理失败: {str(e)}")
# endregion


@legacy_router.post("/startTraining")
async def start_training(train: TrainParameter):
    redis = _legacy_redis
    results = []
    project_name_list = set()
    training_list = set()
    training_record_list = set()
    id_list = set()

    if train.modelType == "2":
        for item in train.trainRecords:
            redis_key = START_TRAINING_PX + item.trainingRecordListId
            redis.set(redis_key, "begin", 3600)
            project_name = f"{item.trainingRecordListId}_{train.modelType}"
            action_name = f"{item.actionId}_{train.modelType}"
            project_name_list.add(project_name)
            training_list.add(item.trainingListId)
            training_record_list.add(item.trainingRecordListId)
            task_manager.start_task(
                train_task_yolo, 0, item.trainingRecordListId, train,
                item.trainingListId, item.trainingRecordListId, item.trainingRecordListId,
                action_name, item.actionId, redis
            )
            results.append({
                "id": id_list,
                "trainingRecordListId": training_record_list,
                "trainingListId": training_list,
                "project_name": project_name_list
            })

    elif train.modelType == "1":
        for itemcnn in train.trainRecords:
            redis_key = START_TRAINING_PX + itemcnn.trainingRecordListId
            redis.set(redis_key, "begin", 3600)
            project_name = f"{itemcnn.trainingRecordListId}_{train.modelType}"
            action_name = f"{itemcnn.actionId}_{train.modelType}"
            project_name_list.add(project_name)
            training_list.add(itemcnn.trainingListId)
            training_record_list.add(itemcnn.trainingRecordListId)
            trainConfig = maskRcnnTrainConfig(
                itemcnn.actionId, itemcnn.trainingRecordListId,
                train.modelType, train.modelConfig.epochs,
                train.modelConfig.batch, train.modelConfig.lr,
                False, train.modelConfig.momentum, train.modelConfig.weightDecay, False, 0
            )
            task_manager.start_task(train_task_maskrcnn, 0, itemcnn.trainingRecordListId, trainConfig, redis)
            results.append({
                "id": id_list,
                "trainingRecordListId": training_record_list,
                "trainingListId": training_list,
                "project_name": project_name_list
            })
    return ajaxReturnYes(results)


@legacy_router.get("/startVideoTrain")
async def start_video_train():
    redis = _legacy_redis
    redis.set(START_TRAINING_PX + "video", "begin", 3600)
    redis.set(START_TRAINING_PX + "video_epoch", "0", 3600)
    new_id = task_manager.start_task(train_task_nn, 0, None, redis)
    return ajaxReturnYes(new_id)


@legacy_router.get("/stopVideoTrain")
async def stop_video_train():
    redis = _legacy_redis
    redis.set(START_TRAINING_PX + "video", "end")
    return ajaxReturnYes("done")


@legacy_router.get("/delModel/{actionId}")
def del_model(actionId: str):
    modelFilePath_CNN = DEDUCTION_MODEL_PATH + "/" + actionId + "_1.pth"
    modelFilePath_YOLO = DEDUCTION_MODEL_PATH + "/" + actionId + "_2.pt"
    if os.path.exists(modelFilePath_CNN):
        os.remove(modelFilePath_CNN)
    if os.path.exists(modelFilePath_YOLO):
        os.remove(modelFilePath_YOLO)
    return ajaxReturnYes(None)


@legacy_router.post("/mixVideos")
async def mix_videos_action(videos: list[str]):
    # 第一个视频不截断头，最后一个视频不截断尾
    print(videos)
    if len(videos) <= 1:
        return ajaxReturnYes("")
    else:
        firstVideo = videos[0].split("/")[len(videos[0].split("/")) - 1]
        videoPath = FILES_PATH + DEDUCTION_PROCESS_LOG_VIDEO_PATH + "/mixVideos_" + firstVideo
        if SaveMixVideos:
            saveMixVideos = SaveMixVideos(videos, videoPath)
            saveMixVideos.MakeFrames()
            saveMixVideos.SaveFramesAsVideo()
        return ajaxReturnYes(DEDUCTION_PROCESS_LOG_VIDEO_PATH + "/mixVideos_" + firstVideo)
