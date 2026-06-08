import time, json
from threading import Event as ThreadPoolEvent
from .common.ajaxResult import getJsonWithOutJwt
from .common.redisManager import RedisManager
from .common.config import *
from .common.threadTaskManager import ThreadTaskManager  # 多线程处理封装
from model import JavaProcessorConfigResult, ProcessorConfig
from process_videoStream import start_process_video_stream

task_manager = ThreadTaskManager(max_workers=20, thread_name_prefix="exService")
redis = RedisManager()


# def run_process(stop_event: ThreadPoolEvent, config, redis: RedisManager):
#     """启动单个推理"""
#     try:
#         if stop_event.is_set():
#             print("推理停止")
#             return
#         print("启动单个推理")
#         time.sleep(20)
#     except Exception as e:
#         raise RuntimeError(f"推理启动失败: {str(e)}")


# region yolo视频流推理
def run_process(stop_event: ThreadPoolEvent, config: ProcessorConfig, redis: RedisManager):
    try:
        if stop_event.is_set():
            print("检测到停止请求，停止推理")
            return
        start_process_video_stream(config, redis)
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise RuntimeError(f"处理失败: {str(e)}")


# endregion


# 额外服务函数
def process_service():
    while True:
        info = f"[EXTRA] Service running at {time.ctime()}"
        print(info)
        # redis.set("extra_service", info, 100)
        # 实际这个configs从java接口获取
        configs = list[ProcessorConfig]
        # print(configs)
        try:
            java_data = getJsonWithOutJwt(JAVA_API_PATH + GET_CAMERA_RETRIEVAL_VO, None)
            data = JavaProcessorConfigResult(**java_data)
            # print(data)
            # print(data.code)
            if data.code == 200:
                configs = data.data
                # print(configs)
                # 循环创建线程，启动多个推理服务  action_id: str, mode_name: str, stream_url: str, rtsp_outPut: str
                for index, config in enumerate(configs):
                    # time.sleep(1)
                    if not hasattr(config, 'id') or config.id == -1:
                        config.id = index
                    statusKey = REDIS_KEY_PROCESS_STATUS + str(config.id)
                    configKey = REDIS_KEY_PROCESS_CONFIG + str(config.id)
                    configCache = redis.get(configKey)
                    taskid = PROCESS_TH_PX + str(index)
                    configStr: str = json.dumps(config.model_dump())
                    if configStr == configCache and config.actionId is not None and task_manager.get_task_info(taskid) is None:
                        # 如果线程中止了，则重新启动
                        print("重启推理1")
                        if index!=-1:
                            taskid = task_manager.start_task(run_process, 0, taskid, config, redis)
                        time.sleep(10)
                            

                    if configStr != configCache:
                        print("变了？")
                        redis.set(configKey, configStr)
                        redis.set(statusKey, "end")
                        time.sleep(10)  # 线程睡10秒，确保推理终止
                        # 其它乱七八糟的缓存也清空
                        # region 初始化缓存名称
                        # 节拍名
                        beat_name_key = REDIS_KEY_BEAT_NAME + str(config.id)
                        redis.delete(beat_name_key)
                        # 节拍最早时间
                        beat_time_key = REDIS_KEY_BEAT_BEGINTIME + str(config.id)
                        redis.delete(beat_time_key)
                        # 节拍本地最早时间
                        beat_local_time_key = REDIS_KEY_BEAT_LOCAL_BEGINTIME + str(config.id)
                        redis.delete(beat_local_time_key)
                        # 标签名
                        label_name_key = REDIS_KEY_LABEL_NAME + str(config.id)
                        redis.delete(label_name_key)
                        # 标签最早时间
                        label_time_key = REDIS_KEY_LABEL_BEGINTIME + str(config.id)
                        redis.delete(label_time_key)
                        # 动作开始时间
                        action_time_key = REDIS_KEY_BEGIN_BEAT_TIME + str(config.id)
                        redis.delete(action_time_key)
                        # 动作序号
                        # action_code_key = REDIS_KEY_ACTION_CODE + str(config.id)
                        # endregion
                        redis.delete(statusKey)
                        if config.actionId is None and task_manager.get_task_info(taskid) is None:
                            print("重启推理2")
                            if index!=-1:
                                taskid = task_manager.start_task(run_process, 0, taskid, config, redis)
                            time.sleep(10)
                # redis.hmset(f"action:{config['actionId']}", config)
        except Exception as e:
            print("错误信息:", str(e))
        print(info)
        time.sleep(PROCESS_ONE_TIME)
