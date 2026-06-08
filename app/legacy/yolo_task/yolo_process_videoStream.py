import threading, time, uuid, os, subprocess, cv2
from collections import deque

import imageio
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from common.ajaxResult import postJsonWithOutJwt
from common.config import *
from common.redisManager import RedisManager
from datetime import datetime

from model import *

# 配置文件夹名称
request_id = uuid.uuid4()
# 将UUID转换为字符串，并去掉连字符
uuid_str = str(request_id).replace('-', '')




class FrameBuffer:
    def __init__(self, delay_frames=100, max_frame_num=1200):
        self.buffer = deque(maxlen=max_frame_num)
        self.delay = delay_frames
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
            self.buffer.append((frame.copy(), formatted_time))  # 关键修改：使用frame.copy()

    def get_last_frame(self):
        """获取延迟处理的帧"""
        if len(self.buffer) > 0:
            return self.buffer.popleft()
        return None

    def get_delay_frame(self):
        """获取延迟处理的帧"""
        if len(self.buffer) > self.delay:
            return self.buffer[len(self.buffer) - self.delay]
        return None

    def clear_buffer(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()

    def get_frames_bytime(self, beginTime, endTime, begin_last_num: int = 12, end_delay_num: int = 6):
        """根据开始时间和结束时间，以及上溯和延后帧数，获取特定帧"""
        with self.lock:
            lenthThis: int = len(self.buffer)
            begin_frame_num: int = -1
            end_frame_num: int = - 1
            for i, item in enumerate(self.buffer):
                frame, formatted_time = item
                if beginTime == formatted_time and begin_frame_num == -1:
                    begin_frame_num = i

                if endTime == formatted_time and end_frame_num == - 1:
                    end_frame_num = i
            if begin_frame_num == -1 or end_frame_num == -1:
                return []
            begin_frame_num = max(0, begin_frame_num - begin_last_num)
            end_frame_num = min(end_frame_num + end_delay_num, lenthThis - 1)
            return [self.buffer[i] for i in range(begin_frame_num, end_frame_num)]


# 保存视频函数
# def save_frames_as_video(frames, output_path, fps, width, height):
#     """将帧序列保存为MP4视频"""
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#     for frame, _ in frames:
#         out.write(frame)
#
#     out.release()
#     return output_path

# 在保存视频函数中，使用原始帧（未处理）
def save_frames_as_video(frames, output_path, fps):
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for (frame, formatted_time) in frames]
    with imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=10
    ) as writer:
        for fram in rgb_frames:
            writer.append_data(fram)
    # """将帧序列保存为MP4视频"""
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # for frame, _ in frames:
    #     out.write(frame)  # 这里使用的是缓冲区存储的原始帧
    # out.release()
    # return output_path


def save_frames_as_image(frames, output_path):
    resized_image = cv2.resize(frames, (960, 540))
    cv2.imwrite(output_path, resized_image)


# region 中文绘制函数
def draw_chinese(image, text, position, color):
    """使用PIL绘制中文文本"""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# endregion


# region 单贞处理函数
# 中文字体配置（确保字体文件存在）
# 加载中文字体
font_path = DEDUCTION_FILE_PATH + "/simhei.ttf"  # 需要下载中文字体文件
font_size = 30
font = ImageFont.truetype(font_path, font_size)
# 自定义颜色方案
color_palette = {
    # 添加更多类别颜色...
}


# 单贞处理
def process_single_frame(single_frame, yolo_model):
    """处理单帧并进行推理"""
    # , stream=True ,half=True,
    results = yolo_model.predict(single_frame, conf=YOLOV8_DEFALT_THRESH, iou=YOLOV8_DEFALT_THRESH)
    detections = []  # 存储检测结果的列表
    for result in results:
        # 绘制分割掩码
        if result.masks is not None:
            for mask in result.masks.xy:
                pts = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.polylines(single_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 遍历每个检测框
        for i, (box, cls, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = yolo_model.names[class_id]
            confidence = conf.item()  # 转换为Python float

            # 获取中文标签（需要自己维护映射字典）
            chinese_labels = {}
            label = chinese_labels.get(class_name, class_name)

            # 存储检测信息 (标签, 置信度, 坐标)
            detections.append({
                "labelName": label,
                "conf": confidence,
                # "bbox": [x1, y1, x2, y2],
                # "class_name": class_name
            })

            # 获取颜色并绘制
            color = color_palette.get(class_name, (0, 255, 255))
            cv2.rectangle(single_frame, (x1, y1), (x2, y2), color, 2)

            # 绘制中文标签（需实现draw_chinese函数）
            text = f"{label} {confidence:.2f}"
            single_frame = draw_chinese(single_frame, text, (x1, y1 - 30), color)

    return single_frame, detections


def process_single_frame_test(single_frame, yolo_model):
    """处理单帧并进行推理"""
    results = yolo_model.predict(single_frame, conf=YOLOV8_DEFALT_THRESH, iou=YOLOV8_DEFALT_THRESH)
    for result in results:
        # 绘制分割掩码
        if result.masks is not None:
            for mask in result.masks.xy:
                # 将分割点转换为整数格式
                pts = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.polylines(single_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 绘制检测框和标签
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = yolo_model.names[class_id]

            # 获取中文标签（需要自己维护映射字典）
            chinese_labels = {}
            label = chinese_labels.get(class_name, class_name)

            # 获取颜色
            color = color_palette.get(class_name, (0, 255, 255))

            # 绘制检测框
            cv2.rectangle(single_frame, (x1, y1), (x2, y2), color, 2)

            # 绘制中文标签
            text = f"{label} {result.boxes.conf[0]:.2f}"
            single_frame = draw_chinese(single_frame, text, (x1, y1 - 30), color)

    return single_frame


# endregion


# region 初始化ffmpeg推流服务
# 推流地址
# rtsp_output = "rtsp://localhost:8554/video_stream"
# 推流视频分辨率
output_width = 640
output_height = 360
# 推流视频贞率
fps = 6


# 初始化ffmpeg进程
def init_ffmpeg(rtsp_outPut: str):
    command = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{output_width}x{output_height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'rtsp',
        rtsp_outPut
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


# endregion


# region  获取节拍名称
# beatList 对象集合
# target_label 标签名称
def find_beat_name_by_label(beats, labelName):
    for beat in beats:
        for label in beat.labelList:
            if label.labelName == labelName:
                return beat.beatName
    return None  # 如果没找到返回 None


# endregion


# 启动推理
def start_process_video_stream(config: ProcessorConfig, mode_name: str, redis: RedisManager):
    # 创建视频保存输出目录
    output_dir = FILES_PATH + DEDUCTION_PROCESS_LOG_VIDEO_PATH
    os.makedirs(output_dir, exist_ok=True)
    # region 初始化缓存名称
    # 节拍名
    beat_name_key = REDIS_KEY_BEAT_NAME + config.actionId
    # 节拍最早时间
    beat_time_key = REDIS_KEY_BEAT_BEGINTIME + config.actionId
    # 节拍本地最早时间
    beat_local_time_key = REDIS_KEY_BEAT_LOCAL_BEGINTIME + config.actionId
    # 标签名
    label_name_key = REDIS_KEY_LABEL_NAME + config.actionId
    # 标签最早时间
    label_time_key = REDIS_KEY_LABEL_BEGINTIME + config.actionId
    # 动作开始时间
    action_time_key = REDIS_KEY_BEGIN_BEAT_TIME + config.actionId
    # endregion

    # region 初始化节拍列表  开始和结束节拍
    beat_list: list[BeatInfo] = config.beatList
    # 开始节拍
    begin_beat_name = beat_list[0].beatName
    # 结束节拍
    last_beat_name = beat_list[-1].beatName
    # endregion

    # 加载训练权重文件
    model = YOLO(DEDUCTION_MODEL_PATH + '/' + mode_name + '.pt')
    # 加载得视频流url或者视频文件得路劲
    video_url = 'D:/WuJianPing/company_project/gr_dk/fjy/截取工艺视频/3_output_2.mp4'

    # 初始化视频流或者视频文件
    cap = cv2.VideoCapture(config.rtspurl)
    #  获取视频帧的分辨率(高和宽)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取视频帧的帧率
    frame_fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频保存器(将推流后得视频保存指定目录)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用'XVID'
    # out = cv2.VideoWriter(
    #     os.path.join(output_dir, uuid_str + '.mp4'),
    #     fourcc,
    #     # 10,
    #     frame_fps if frame_fps != 0 else 30,  # 如果获取不到fps则默认30
    #     (frame_width, frame_height)
    # )

    # 初始化ffmpeg推流
    ffmpeg_process = init_ffmpeg(config.ffurl)
    buffer = FrameBuffer(delay_frames=9, max_frame_num=1200)
    i: int = 0
    frame_count = 0
    # 动作判定结果  1：OK 2：NG，3：未开始（默认3）
    action_status = "3"
    # 叠加动作日志格式:{节拍1}:{节拍1日志};{节拍2}:{节拍2日志}------记得是叠加
    action_logs = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 处理当前帧
        # processed_frame = process_single_frame(frame, model)
        if i % 5 == 1:
            buffer.add_frame(frame)
            delayed_frame = buffer.get_delay_frame()
            if delayed_frame is not None:
                frame_data, timestamp = delayed_frame

                # 创建副本进行处理
                process_frame = frame_data.copy()  # 关键修改：创建处理副本
                processed_frame, detection_list = process_single_frame(process_frame, model)

                # 创建保存推理贞图片目录
                # 获取当前日期时间 格式化为yyyymmdd字符串
                current_datetime = datetime.now()
                formatted_date = current_datetime.strftime("%Y%m%d")
                # 保存的目标路径
                folder_path = FILES_PATH + DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date
                # 判断是否存在目录
                if not os.path.exists(folder_path):
                    # 不存在自动创建多级目录
                    os.makedirs(folder_path)

                # 构造贞图片输出文件名
                new_id = uuid.uuid4()
                # 将UUID转换为字符串，并去掉连字符
                frame_name = f"{str(new_id).replace('-', '')}.jpg"
                output_path = os.path.join(folder_path, frame_name)

                logs_content = None
                log_lv = None
                log_type = None
                ng_value = None
                action_long = None
                video_url = None
                beat_begin_time = None
                action_begin_time = None

                is_java_send = True
                is_have_third = False  # 是否触发逻辑3
                if redis is not None:
                    # 获取缓存数据
                    # 节拍名
                    redis_beat_name = redis.get(beat_name_key)
                    # 节拍最早时间
                    redis_beat_time = redis.get(beat_time_key)
                    # 标签名
                    redis_label_name = redis.get(label_name_key)
                    # 标签最早时间
                    redis_label_time = redis.get(label_time_key)
                    # 动作开始时间
                    redis_action_time = redis.get(action_time_key)

                    # 逻辑3  动作开始时间不为空  +  当前时间 - 动作开始时间 > 2分钟,如果触发了逻辑3，后面就不需要走了
                    if redis_action_time is not None:
                        # 字符串转datetime对象
                        time1 = datetime.strptime(redis_action_time, "%Y-%m-%d %H:%M:%S")
                        time2 = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        # 计算秒数差
                        action_long = (time2 - time1).total_seconds()
                        if action_long > 120:
                            redis.delete(action_time_key)
                            is_have_third = True
                            log_type = '2'
                            action_status = "2"
                            # 叠加日志
                            if action_logs is None:
                                action_logs = f'自动判定:动作{config.actionName}超过2分钟未结束，严重超时'
                            else:
                                action_logs = action_logs + f';自动判定:动作{config.actionName}超过2分钟未结束，严重超时'
                            # 保存视频调用接口
                            action_frames = buffer.get_frames_bytime(redis_beat_time, timestamp)
                            if action_frames:
                                # 生成唯一视频文件名
                                # 构造贞图片输出文件名
                                video_action_id = uuid.uuid4()
                                # 将UUID转换为字符串，并去掉连字符
                                video_name = f"{str(video_action_id).replace('-', '')}_{config.actionId}.mp4"
                                video_path = os.path.join(output_dir, video_name)
                                # 保存视频
                                # save_frames_as_video(
                                #     frames=action_frames,
                                #     output_path=video_path,
                                #     fps=fps,
                                #     width=frame_width,
                                #     height=frame_height
                                # )
                                # 线程保存视频
                                threading.Thread(target=save_frames_as_video,
                                                 args=(action_frames, video_path, fps)).start()
                                threading.Thread(target=postJsonWithOutJwt,
                                                 args=(JAVA_API_PATH + SEND_PROCESS_RESULT_TO_JAVA, {
                                                     "actionId": config.actionId,
                                                     "actionName": config.actionName,
                                                     "actionStatus": action_status,
                                                     "actionLogs": action_logs,
                                                     "actionLong": action_long,
                                                     "logType": '2',
                                                     "videoPath": DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + video_name,
                                                     "actionBeginTime": redis_action_time,
                                                     "endTime": timestamp,
                                                 })).start()

                    if not is_have_third:
                        # 遍历检测结果 获取标签名称和匹配率
                        # 回传数据到java端处理
                        # 确保detection_list是列表类型（防止返回None）
                        if detection_list is None or len(detection_list) == 0:
                            print("未检测到任何对象")
                            # 当前节拍为空   +  缓存节拍为最后节拍  +   当前时间 - 缓存节拍开始时间>10S
                            # 清空缓存节拍 + 清空缓存动作开始时间 + 重置上面两个变量
                            if redis_beat_name == last_beat_name:
                                if redis_beat_time is not None:
                                    # 字符串转datetime对象
                                    time1 = datetime.strptime(redis_beat_time, "%Y-%m-%d %H:%M:%S")
                                    time2 = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                    # 计算秒数差
                                    beat_long = (time2 - time1).total_seconds()
                                    if beat_long > 10:
                                        redis.delete(beat_name_key)
                                        redis.delete(beat_time_key)
                                        redis.delete(action_time_key)
                                        action_status = '3'
                                        action_logs = None
                        else:
                            # 业务处理
                            result_map = {}
                            for obj in detection_list:
                                # 可选：添加对象有效性检查
                                if not obj or 'labelName' not in obj or 'conf' not in obj:
                                    print("遇到无效检测结果，跳过")
                                    continue
                                key = obj['labelName']
                                value = f"{obj['conf']:.2f}"
                                # 如果 key 已存在，则保留最大值；否则直接插入
                                if key in result_map:
                                    if value > result_map[key]:
                                        result_map[key] = value
                                else:
                                    result_map[key] = value
                            # 获取值最大的 key
                            max_label_key = max(result_map, key=result_map.get)
                            # 用key获取值
                            max_label_value = result_map[max_label_key]
                            if redis is not None:
                                # if redis_label_name == max_label_key:
                                #     is_java_send = False
                                #
                                # if redis_label_name is None and redis_label_time is None:
                                #     # 当缓存没有标签的时候 缓存进来标签名称和标签开始时间
                                #     redis.set(label_name_key, max_label_key)
                                #     redis.set(label_time_key, timestamp)
                                # else:
                                #     # 标签不一样的时候更新缓存
                                #     if redis_label_name != max_label_key:
                                #         redis.set(label_name_key, max_label_key)
                                #         redis.set(label_time_key, timestamp)

                                # 根据标签获取节拍名
                                list_beat_name = find_beat_name_by_label(beat_list, max_label_key)
                                if redis_beat_name is None and redis_beat_time is None:
                                    # 缓存获取的节拍和节拍开始时间都为空代表第一次进入
                                    # 标签获取的节拍等于开始节拍
                                    if list_beat_name == begin_beat_name:
                                        print('是开始节拍')
                                        # 记录缓存动作开始时间
                                        redis.set(action_time_key, timestamp)
                                        # 记录日志
                                        logs_content = f'当前节拍: {list_beat_name},标签名称: {max_label_key},匹配率:{max_label_value}'
                                        log_lv = '1'
                                        action_status = '1'
                                    else:
                                        print('不是开始节拍')
                                        # 记录日志
                                        logs_content = f'无法找到起始动作,判定NG'
                                        log_lv = '3'
                                        action_status = '2'
                                    # 更新缓存节拍名和节拍时间
                                    redis.set(beat_name_key, list_beat_name)
                                    redis.set(beat_time_key, timestamp)
                                else:
                                    # 当前节拍不是缓存节拍的时候
                                    # 两种情况一种是结束结束节拍一种是是不是下一个节拍
                                    if list_beat_name != redis_beat_name:
                                        print('当前节拍和缓存节拍不一致')
                                        # 当缓存节拍和当前节拍不一致的时候说明节拍结束写入 beat_logs
                                        # logType: 1:节拍
                                        # beatBeginTime: 节拍开始时间等同于 开始获取的redis的节拍时间
                                        log_type = '1'
                                        beat_begin_time = redis_beat_time
                                        # 保存节拍视频调用接口
                                        beat_frames = buffer.get_frames_bytime(redis_beat_time, timestamp)
                                        if beat_frames:
                                            # 生成唯一视频文件名
                                            # 构造贞图片输出文件名
                                            video_beat_id = uuid.uuid4()
                                            # 将UUID转换为字符串，并去掉连字符
                                            video_beat_name = f"{str(video_beat_id).replace('-', '')}_{config.actionId}.mp4"
                                            video_beat_path = os.path.join(output_dir, video_beat_name)
                                            video_url = DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + video_beat_name
                                            # 保存视频
                                            threading.Thread(target=save_frames_as_video,
                                                             args=(beat_frames, video_beat_path, fps)).start()
                                            # save_frames_as_video(
                                            #     frames=beat_frames,
                                            #     output_path=video_beat_path,
                                            #     fps=fps,
                                            #     width=frame_width,
                                            #     height=frame_height
                                            # )

                                        # 动作开始时间不为空 + 当前节拍为最后节拍 == == == >
                                        # 计算动作时间 + 清空缓存动作开始时间 + 重置上面两个变量 == == == = > 最后统一调接口
                                        if redis_action_time is not None and list_beat_name == last_beat_name:
                                            log_type = '2'
                                            action_status = '3'
                                            action_logs = None
                                            # 字符串转datetime对象
                                            action_begin_time = redis_action_time
                                            time1 = datetime.strptime(redis_action_time, "%Y-%m-%d %H:%M:%S")
                                            time2 = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                            # 计算秒数差
                                            action_long = (time2 - time1).total_seconds()
                                            redis.delete(action_time_key)
                                            # 保存节拍视频调用接口
                                            end_video_frames = buffer.get_frames_bytime(redis_action_time, timestamp)
                                            if end_video_frames:
                                                # 生成唯一视频文件名
                                                # 构造贞图片输出文件名
                                                end_video_beat_id = uuid.uuid4()
                                                # 将UUID转换为字符串，并去掉连字符
                                                end_video_name = f"{str(end_video_beat_id).replace('-', '')}_{config.actionId}.mp4"
                                                end_video_path = os.path.join(output_dir, end_video_name)
                                                video_url = DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + end_video_name
                                                # 保存视频
                                                threading.Thread(target=save_frames_as_video,
                                                                 args=(end_video_frames, end_video_path, fps)).start()
                                                # save_frames_as_video(
                                                #     frames=end_video_frames,
                                                #     output_path=end_video_path,
                                                #     fps=fps,
                                                #     width=frame_width,
                                                #     height=frame_height
                                                # )

                                        #  缓存节拍是结束节拍的时候
                                        if redis_beat_name == last_beat_name:
                                            print('缓存节拍是结束节拍')
                                            # 判断当前节拍是不是开始节拍
                                            if list_beat_name == begin_beat_name:
                                                print('当前节拍是开始节拍')
                                                # 缓存节拍是开始节拍
                                                redis.set(action_time_key, timestamp)
                                                action_begin_time = timestamp
                                                # 记录日志
                                                logs_content = f'当前节拍: {list_beat_name},标签名称: {max_label_key},匹配率:{max_label_value}'
                                                log_lv = '1'
                                                action_status = '1'
                                            else:
                                                print("不是开始节拍")
                                                # 记录日志
                                                logs_content = f'未捕捉起始节拍: {begin_beat_name},直接执行了: {list_beat_name}节拍,工序混乱NG'
                                                log_lv = '3'
                                                action_status = '2'
                                        else:
                                            print('缓存节拍不是结束节拍')
                                            # 缓存节拍不是结束节拍的时候
                                            # 判断是不是下一个节拍
                                            # 初始化结果
                                            is_next = False
                                            # 遍历列表（从第一个元素到倒数第二个元素）
                                            for i in range(len(beat_list) - 1):
                                                # 检查当前节拍是否为缓存的下一个节拍
                                                if beat_list[i].beatName == redis_beat_name and \
                                                        beat_list[i + 1].beatName == list_beat_name:
                                                    is_next = True
                                                    break
                                            if is_next:
                                                print("是下一节拍")
                                                # 记录日志
                                                logs_content = f'当前节拍: {list_beat_name},标签名称: {max_label_key},匹配率:{max_label_value}'
                                                log_lv = '1'
                                                action_status = '1'
                                            else:
                                                print("不是下一节拍")
                                                # 记录日志
                                                logs_content = f'当前节拍:{list_beat_name},不是节拍:{redis_beat_name}的下一个节拍,工序混乱NG'
                                                log_lv = '3'
                                                action_status = '2'
                                        # 更新缓存节拍名称和节拍开始时间
                                        redis.set(beat_name_key, list_beat_name)
                                        redis.set(beat_time_key, timestamp)
                                    else:
                                        is_java_send = False

                                # 叠加日志
                                if action_logs is None:
                                    action_logs = f'{list_beat_name}:{logs_content}'
                                else:
                                    action_logs = action_logs + f';{list_beat_name}:{logs_content}'

                                # 查找匹配的 ID
                                # beat_id = [beat.id for beat in beat_list if beat.beatName == list_beat_name]
                                beat_id = next((beat.id for beat in beat_list if beat.beatName == list_beat_name), None)
                                if is_java_send:
                                    threading.Thread(target=postJsonWithOutJwt,
                                                     args=(JAVA_API_PATH + SEND_PROCESS_RESULT_TO_JAVA, {
                                                         "actionId": config.actionId,
                                                         "camerasId": config.camerasId,
                                                         "beatId": beat_id,
                                                         "beatName": list_beat_name,
                                                         "labels": max_label_key,
                                                         "logsContent": logs_content,
                                                         "logLv": log_lv,
                                                         "imagePath": DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date + '/' + frame_name,
                                                         "actionTime": timestamp,
                                                         "logType": log_type,
                                                         "videoPath": video_url,
                                                         "beatBeginTime": beat_begin_time,
                                                         "actionBeginTime": action_begin_time,
                                                         "endTime": timestamp,
                                                         "actionStatus": action_status,
                                                         "actionLogs": action_logs,
                                                     })).start()
                                    # 保存帧为图片
                                    threading.Thread(target=save_frames_as_image,
                                                     args=(processed_frame, output_path)).start()
                                    # resized_image = cv2.resize(processed_frame, (960, 540))
                                    # cv2.imwrite(output_path, resized_image)
                        # 调整分辨率并推流
                        resized_frame = cv2.resize(processed_frame, (output_width, output_height))
                        try:
                            # 推流到ffmpeg
                            ffmpeg_process.stdin.write(resized_frame.tobytes())
                        except Exception as e:
                            print(f"推流失败: {str(e)}")
                            # 重启推流进程
                            ffmpeg_process = init_ffmpeg(config.ffurl)
        else:
            print("drop frame")
        i = i + 1
        if i == 5:
            i = 0
    # 释放资源
    cap.release()
    if ffmpeg_process.poll() is None:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    ls: list[BeatInfo] = [
        {
            "id": "28fbffb01a9f4890b1d579e7f38a7200",
            "beatName": "整理排线",
            "labelList": [
                {
                    "id": "6a5fca43fd1e41c7bad144c0af714b60",
                    "labelName": "整理排线1",
                    "isMain": "1"
                },
                {
                    "id": "61e9ee5870b842c696df8865fcd80259",
                    "labelName": "整理排线2",
                    "isMain": "0"
                }
            ]
        },
        {
            "id": "650de083ea614a1789a76276d8b82cc6",
            "beatName": "取屏蔽罩",
            "labelList": [
                {
                    "id": "0f137e1d75ec4252ba6a92cda95786ad",
                    "labelName": "取屏蔽罩到工件",
                    "isMain": "1"
                },
                {
                    "id": "0762164225e0454092829728dd5d8c67",
                    "labelName": "放屏蔽罩到工件",
                    "isMain": "0"
                }
            ]
        },
        {
            "id": "f61869a9c4d745cd8286ccb6138287d6",
            "beatName": "取主机板",
            "labelList": [
                {
                    "id": "601be347329141ac8afd1264247d8c89",
                    "labelName": "取主机板到工件",
                    "isMain": "1"
                },
                {
                    "id": "3b49b375152248eaa5f0ccd8140020f3",
                    "labelName": "放主机板到工件",
                    "isMain": "0"
                }
            ]
        },
        {
            "id": "a9500a06ae364a37aae3772118ad400d",
            "beatName": "涂油",
            "labelList": [
                {
                    "id": "0bdbb5d654e34053af34f7c71ca5d499",
                    "labelName": "涂油放屏蔽罩",
                    "isMain": "1"
                },
                {
                    "id": "e5a060e688624cada6dd933ac9e8db97",
                    "labelName": "涂油放主机板",
                    "isMain": "0"
                },
                {
                    "id": "13ec5c3f4697491b97c5522a4988b7e9",
                    "labelName": "涂油扫主板码3",
                    "isMain": "0"
                }
            ]
        },
        {
            "id": "abd209aa94d646e9847c41f1104796b3",
            "beatName": "打螺丝",
            "labelList": [
                {
                    "id": "880986952ad642e4b51b38b76f02d5f5",
                    "labelName": "放置打螺丝固定盖板",
                    "isMain": "1"
                },
                {
                    "id": "95cac7465f5441f7bb69444001e1a143",
                    "labelName": "打螺丝",
                    "isMain": "0"
                }
            ]
        },
        {
            "id": "1524ed52936a4c7d8a20d76c5b855a14",
            "beatName": "插wifi线",
            "labelList": [
                {
                    "id": "2206647145964affbb1928404f253137",
                    "labelName": "毛刷枪清理1",
                    "isMain": "0"
                },
                {
                    "id": "79a52a4736e04f2590fd4399691429ae",
                    "labelName": "毛刷枪清理2",
                    "isMain": "0"
                },
                {
                    "id": "e8ae5ebc7a2a43a9bd74e1d5fe445cb2",
                    "labelName": "插wifi线1",
                    "isMain": "1"
                },
                {
                    "id": "440c24ed4706470da0842999ee7be797",
                    "labelName": "插wifi线2",
                    "isMain": "0"
                },
                {
                    "id": "40a54190937e40f4b9448fe5cf1b201b",
                    "labelName": "插wifi线3",
                    "isMain": "0"
                }
            ]
        },
        {
            "id": "82b4dfb3589a42de9ac591b6ee527f52",
            "beatName": "打点1",
            "labelList": [
                {
                    "id": "f0923dc449e54f02be1931ac9654ccc3",
                    "labelName": "打点",
                    "isMain": "1"
                }
            ]
        },
        {
            "id": "5ec0f8a10ffb4726b088dcb92110b8df",
            "beatName": "打点2",
            "labelList": [
                {
                    "id": "0a6b399237da4425a5bd9ae816b51983",
                    "labelName": "打点划线1",
                    "isMain": "1"
                },
                {
                    "id": "381d794144ab4b82af4a67738240a84c",
                    "labelName": "打点划线2",
                    "isMain": "0"
                },
                {
                    "id": "2ed28094d346458097679c5a1e1370fe",
                    "labelName": "打点划线3",
                    "isMain": "0"
                },
                {
                    "id": "81b2a51875b24268b5818586b2ab5467",
                    "labelName": "打点划线4",
                    "isMain": "0"
                }
            ]
        },
        {
            "id": "627930643df2465a945b30db7d96adb4",
            "beatName": "结束扫码",
            "labelList": [
                {
                    "id": "52d6fc08ca044a0ca6111869fee94488",
                    "labelName": "拿标签码",
                    "isMain": "1"
                },
                {
                    "id": "39c568b0735649bba01d82e7b088cc29",
                    "labelName": "放标签码",
                    "isMain": "0"
                },
                {
                    "id": "eeedf1c15893404391adc2ccf91ebcc0",
                    "labelName": "结束扫主机板码3",
                    "isMain": "0"
                },
                {
                    "id": "ead7c00fe2d74f2692ffcf4f3972f2b8",
                    "labelName": "结束扫固定条形码3",
                    "isMain": "0"
                },
                {
                    "id": "0198e27c719b4932a07c42699c1f3434",
                    "labelName": "结束扫标签码3",
                    "isMain": "0"
                }
            ]
        }
    ]

    target_beat_name = "取屏蔽罩"

    # # 查找匹配的 ID 650de083ea614a1789a76276d8b82cc6
    # matches = [beat['id'] for beat in ls if beat['beatName'] == target_beat_name]
    # print(matches)

    # region  获取对象数组最后一个对象和第一个对象
    # # 开始节拍
    # begin_beat_name = ls[0]["beatName"]
    # # 结束节拍
    # last_beat_name = ls[-1]["beatName"]
    # endregion

    # region  判断某个节拍是否为是缓存节拍的下一个节拍
    # redis_name = '涂油'
    # # 打螺丝
    # be_name = '结束扫码'
    # is_to_next = False
    # for i in range(len(ls) - 1):
    #     # 检查当前元素是否为"取屏蔽罩"，且下一个元素是否为"取主机板"
    #     if ls[i]["beatName"] == redis_name and ls[i + 1]["beatName"] == be_name:
    #         print("当前元素为: %s, 下一个元素为: %s" % (ls[i]["beatName"], ls[i + 1]["beatName"]))
    #         is_to_next = True
    #         break
    #
    # if is_to_next:
    #     print("进入下一节拍")
    # else:
    #     print("未进入下一节拍")
    #
    # print(is_to_next)
    # endregion

    # region 用当前标签获取数组对应标签的节拍
    # print(ls)
    # result = find_beat_name_by_label(ls, "取屏蔽罩到工件")
    # print(f" beatName 是: {result}")
    # endregion

    # region 将数组对象放入到map
    # result_map = {}
    # order = 0
    # for beat in ls:
    #     order += 1
    #     key = beat['beatName']
    #     value = order
    #     result_map[key] = value
    #
    # print(result_map)
    # print(order)
    # max_key = max(result_map, key=result_map.get)
    # min_key = min(result_map, key=result_map.get)
    # print(max_key)
    # print(min_key)
    # endregion

    # detections = (
    #     {
    #         "labelName": '整理排线2',
    #         "conf": 0.9931,
    #     },
    #     {
    #         "labelName": '整理排线1',
    #         "conf": 0.9991,
    #     }
    # )
    # result_map = {}
    # for obj in detections:
    #     # 可选：添加对象有效性检查
    #     if not obj or 'labelName' not in obj or 'conf' not in obj:
    #         print("遇到无效检测结果，跳过")
    #         continue
    #
    #     print(f"标签: {obj['labelName']}, 置信度: {obj['conf']:.2f}")
    #     key = obj['labelName']
    #     value = f"{obj['conf']:.2f}"
    #     # 如果 key 已存在，则保留最大值；否则直接插入
    #     if key in result_map:
    #         if value > result_map[key]:
    #             result_map[key] = value
    #     else:
    #         result_map[key] = value
    #
    # # 获取值最大的 key
    # max_key = max(result_map, key=result_map.get)
    # print(max_key)

    redis = RedisManager()
    configs = list[ProcessorConfig]
    da = {
        "code": 200,
        "msg": "操作成功",
        "data": [
            {
                "actionId": "edcf584cf48c48e7a065a5f296420ce6",
                "actionName": "三工艺",
                "actionType": "2",
                "camerasId": "2",
                "camerasName": "工艺三模拟",
                "camerasStatus": "1",
                "rtspurl": "rtsp://192.168.1.188:8554/craft3",
                "ffurl": "rtsp://localhost:8554/video_stream",
                "httpurl": "http://192.168.1.188:8888/video_stream",
                "workstationId": "e85d23c15a9d4cfa9d23fd88f60dfbc3",
                "workstationName": "测试12",
                "beatList": [
                    {
                        "id": "28fbffb01a9f4890b1d579e7f38a7200",
                        "beatName": "整理排线",
                        "labelList": [
                            {
                                "id": "6a5fca43fd1e41c7bad144c0af714b60",
                                "labelName": "整理排线1",
                                "isMain": "1"
                            },
                            {
                                "id": "61e9ee5870b842c696df8865fcd80259",
                                "labelName": "整理排线2",
                                "isMain": "0"
                            }
                        ]
                    },
                    {
                        "id": "650de083ea614a1789a76276d8b82cc6",
                        "beatName": "取屏蔽罩",
                        "labelList": [
                            {
                                "id": "0f137e1d75ec4252ba6a92cda95786ad",
                                "labelName": "取屏蔽罩到工件",
                                "isMain": "1"
                            },
                            {
                                "id": "0762164225e0454092829728dd5d8c67",
                                "labelName": "放屏蔽罩到工件",
                                "isMain": "0"
                            }
                        ]
                    },
                    {
                        "id": "f61869a9c4d745cd8286ccb6138287d6",
                        "beatName": "取主机板",
                        "labelList": [
                            {
                                "id": "601be347329141ac8afd1264247d8c89",
                                "labelName": "取主机板到工件",
                                "isMain": "1"
                            },
                            {
                                "id": "3b49b375152248eaa5f0ccd8140020f3",
                                "labelName": "放主机板到工件",
                                "isMain": "0"
                            }
                        ]
                    },
                    {
                        "id": "a9500a06ae364a37aae3772118ad400d",
                        "beatName": "涂油",
                        "labelList": [
                            {
                                "id": "0bdbb5d654e34053af34f7c71ca5d499",
                                "labelName": "涂油放屏蔽罩",
                                "isMain": "1"
                            },
                            {
                                "id": "e5a060e688624cada6dd933ac9e8db97",
                                "labelName": "涂油放主机板",
                                "isMain": "0"
                            },
                            {
                                "id": "13ec5c3f4697491b97c5522a4988b7e9",
                                "labelName": "涂油扫主板码3",
                                "isMain": "0"
                            }
                        ]
                    },
                    {
                        "id": "abd209aa94d646e9847c41f1104796b3",
                        "beatName": "打螺丝",
                        "labelList": [
                            {
                                "id": "880986952ad642e4b51b38b76f02d5f5",
                                "labelName": "放置打螺丝固定盖板",
                                "isMain": "1"
                            },
                            {
                                "id": "95cac7465f5441f7bb69444001e1a143",
                                "labelName": "打螺丝",
                                "isMain": "0"
                            }
                        ]
                    },
                    {
                        "id": "1524ed52936a4c7d8a20d76c5b855a14",
                        "beatName": "插wifi线",
                        "labelList": [
                            {
                                "id": "2206647145964affbb1928404f253137",
                                "labelName": "毛刷枪清理1",
                                "isMain": "0"
                            },
                            {
                                "id": "79a52a4736e04f2590fd4399691429ae",
                                "labelName": "毛刷枪清理2",
                                "isMain": "0"
                            },
                            {
                                "id": "e8ae5ebc7a2a43a9bd74e1d5fe445cb2",
                                "labelName": "插wifi线1",
                                "isMain": "1"
                            },
                            {
                                "id": "440c24ed4706470da0842999ee7be797",
                                "labelName": "插wifi线2",
                                "isMain": "0"
                            },
                            {
                                "id": "40a54190937e40f4b9448fe5cf1b201b",
                                "labelName": "插wifi线3",
                                "isMain": "0"
                            }
                        ]
                    },
                    {
                        "id": "82b4dfb3589a42de9ac591b6ee527f52",
                        "beatName": "打点1",
                        "labelList": [
                            {
                                "id": "f0923dc449e54f02be1931ac9654ccc3",
                                "labelName": "打点",
                                "isMain": "1"
                            }
                        ]
                    },
                    {
                        "id": "5ec0f8a10ffb4726b088dcb92110b8df",
                        "beatName": "打点2",
                        "labelList": [
                            {
                                "id": "0a6b399237da4425a5bd9ae816b51983",
                                "labelName": "打点划线1",
                                "isMain": "1"
                            },
                            {
                                "id": "381d794144ab4b82af4a67738240a84c",
                                "labelName": "打点划线2",
                                "isMain": "0"
                            },
                            {
                                "id": "2ed28094d346458097679c5a1e1370fe",
                                "labelName": "打点划线3",
                                "isMain": "0"
                            },
                            {
                                "id": "81b2a51875b24268b5818586b2ab5467",
                                "labelName": "打点划线4",
                                "isMain": "0"
                            }
                        ]
                    },
                    {
                        "id": "627930643df2465a945b30db7d96adb4",
                        "beatName": "结束扫码",
                        "labelList": [
                            {
                                "id": "52d6fc08ca044a0ca6111869fee94488",
                                "labelName": "拿标签码",
                                "isMain": "1"
                            },
                            {
                                "id": "39c568b0735649bba01d82e7b088cc29",
                                "labelName": "放标签码",
                                "isMain": "0"
                            },
                            {
                                "id": "eeedf1c15893404391adc2ccf91ebcc0",
                                "labelName": "结束扫主机板码3",
                                "isMain": "0"
                            },
                            {
                                "id": "ead7c00fe2d74f2692ffcf4f3972f2b8",
                                "labelName": "结束扫固定条形码3",
                                "isMain": "0"
                            },
                            {
                                "id": "0198e27c719b4932a07c42699c1f3434",
                                "labelName": "结束扫标签码3",
                                "isMain": "0"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    data = JavaProcessorConfigResult(**da)
    print(data)
    if data.code == 200:
        configs = data.data
        for con in configs:
            taskid = PROCESS_TH_PX + con.actionId
            model_name = con.actionId + '_' + con.actionType
            start_process_video_stream(con, model_name, redis)
