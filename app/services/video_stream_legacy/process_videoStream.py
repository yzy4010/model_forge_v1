import platform
import shutil
import threading, time, uuid, os, subprocess, cv2, torch, json, re
from collections import deque

try:
    import imageio  # type: ignore[import]
except Exception:  # 兼容部分环境下 imageio 元数据异常
    imageio = None
from threading import Event as ThreadPoolEvent
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from app.services.video_stream_legacy.common.ajaxResult import postJsonWithOutJwt
from app.services.video_stream_legacy.common.config import *
from app.services.video_stream_legacy.common.redisManager import RedisManager
from datetime import datetime, timedelta

from app.services.video_stream_legacy.common.threadTaskManager import ThreadTaskManager
from app.services.video_stream_legacy.model import *

try:
    from rcn.backbone.resnet50_fpn_model import resnet50_fpn_backbone  # type: ignore[import]
    from rcn.draw_box_utils import draw_objs  # type: ignore[import]
    from rcn.network_files.mask_rcnn import MaskRCNN  # type: ignore[import]
    from torchvision import transforms  # type: ignore[import]
    from rcn.labelAction import masks_to_coco_polygon  # type: ignore[import]
except Exception:
    resnet50_fpn_backbone = None
    draw_objs = None
    MaskRCNN = None
    transforms = None
    masks_to_coco_polygon = None
from queue import Queue

# 配置文件夹名称
request_id = uuid.uuid4()
# 将UUID转换为字符串，并去掉连字符
uuid_str = str(request_id).replace('-', '')
# 中文字体配置（确保字体文件存在）
# 加载中文字体
font_path = DEDUCTION_FILE_PATH + "/simhei.ttf"  # 需要下载中文字体文件
# 推流地址
# rtsp_output = "rtsp://localhost:8554/video_stream"
# 推流视频分辨率
output_width = 640
output_height = 360
# 推流视频分辨率
buffer_frame_width = 768
buffer_frame_height = 432
# 字体大小
font_size = 24
# 边框大小
line_size = 1
# 推流视频贞率
out_fps = 6
# 缓冲区延时帧数（1.5S）
def_delay_frames_num = 5 * out_fps
# 缓冲区最大帧数（保留2分钟）
def_max_frame_num = 125 * out_fps
# 获取时间段内帧向前延申帧数（5秒）
def_begin_last_num = 5 * out_fps
# 获取时间段内帧向后延申帧数（1秒）
def_end_delay_num = 5 * out_fps - 2
# 动作最长时间
max_action_time = 110

font = ImageFont.truetype(font_path, font_size)
# 自定义颜色方案
color_palette = {
    # 添加更多类别颜色...
}

task_manager = ThreadTaskManager(max_workers=30, thread_name_prefix="process")

# ----- 运行平台：自动识别 Windows / Linux / macOS / 其他 -----
_os_sys = platform.system().lower()
if _os_sys == "windows":
    OS_KIND = "windows"
elif _os_sys == "linux":
    OS_KIND = "linux"
elif _os_sys == "darwin":
    OS_KIND = "darwin"
else:
    OS_KIND = _os_sys

IS_WINDOWS = OS_KIND == "windows"
IS_LINUX = OS_KIND == "linux"
IS_MACOS = OS_KIND == "darwin"
# 兼容历史变量名（本文件内多处使用）
osName = OS_KIND


def _ffmpeg_use_nvenc() -> bool:
    """是否使用 NVIDIA NVENC；无 CUDA 或环境关闭时用 CPU libx264（各系统一致）。"""
    if os.getenv("MODEL_FORGE_FFMPEG_NVENC", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    return bool(torch.cuda.is_available())


# h264_nvenc 的 -preset：FFmpeg 4.2（Rocky 等 rpm）只认 ll/fast/hq 等；p1~p7 为较新版本才有。
# 需要可设置环境变量 MODEL_FORGE_FFMPEG_NVENC_PRESET=fast 等覆盖。
FFMPEG_NVENC_PRESET = (
    os.getenv("MODEL_FORGE_FFMPEG_NVENC_PRESET", "ll").strip() or "ll"
)


def _resolve_imageio_ffmpeg_exe() -> None:
    """按系统为 imageio 设置 ffmpeg 可执行文件路径。"""
    if IS_WINDOWS:
        for p in (
            r"C:\Tools\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        ):
            if os.path.isfile(p):
                os.environ["IMAGEIO_FFMPEG_EXE"] = p
                return
        found = shutil.which("ffmpeg")
        if found:
            os.environ["IMAGEIO_FFMPEG_EXE"] = found
        return
    if IS_LINUX:
        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
        return
    if IS_MACOS:
        for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.isfile(p):
                os.environ["IMAGEIO_FFMPEG_EXE"] = p
                return
        found = shutil.which("ffmpeg")
        if found:
            os.environ["IMAGEIO_FFMPEG_EXE"] = found
        return
    found = shutil.which("ffmpeg")
    if found:
        os.environ["IMAGEIO_FFMPEG_EXE"] = found
    else:
        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"


# 设置环境变量启用 FFmpeg 日志
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;verbose"
os.environ["OPENCV_LOG_LEVEL"] = "DEBUG"
_resolve_imageio_ffmpeg_exe()


class FFmpegErrorCatcher:
    def __init__(self):
        self.error_queue = Queue()
        self._original_stderr = os.dup(2)  # 备份原始stderr
        self.r_pipe, self.w_pipe = os.pipe()
        os.dup2(self.w_pipe, 2)  # 重定向stderr到管道

        # 启动监听线程
        self.thread = threading.Thread(target=self._monitor_stderr)
        self.thread.daemon = True
        self.thread.start()

    def _monitor_stderr(self):
        """从管道读取并分析错误消息"""
        with os.fdopen(self.r_pipe) as pipe:
            while True:
                line = pipe.readline()
                if not line: break
                if "error while decoding MB" in line:
                    # 提取错误细节
                    match = re.search(r"error while decoding MB (\d+)\s+(\d+)", line)
                    if match:
                        self.error_queue.put({
                            "type": "h264_mb_error",
                            "mb_x": match.group(1),
                            "mb_y": match.group(2),
                            "message": line.strip()
                        })
                # 可选：将消息打印到控制台
                os.write(self._original_stderr, line.encode())

    def get_error(self):
        """获取队列中的错误"""
        if not self.error_queue.empty():
            return self.error_queue.get()
        return None

    def close(self):
        """恢复 stderr 并关闭 pipe/fd（异常保护）。"""
        try:
            if hasattr(self, "_original_stderr"):
                os.dup2(self._original_stderr, 2)
        except Exception:
            pass
        try:
            if hasattr(self, "w_pipe"):
                os.close(self.w_pipe)
        except Exception:
            pass
        try:
            if hasattr(self, "_original_stderr"):
                os.close(self._original_stderr)
        except Exception:
            pass


# 创建视频保存输出目录
# output_dir = FILES_PATH + DEDUCTION_PROCESS_LOG_VIDEOS_PATH
# os.makedirs(output_dir, exist_ok=True)

# region 帧缓冲区
class FrameBuffer:
    def __init__(self, delay_frames=def_delay_frames_num, max_frame_num=def_max_frame_num):
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
            self.buffer.append((frame, formatted_time))  # 关键修改：使用frame.copy()

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

    def get_frames_bytime(self, beginTime, endTime, begin_last_num: int = def_begin_last_num,
                          end_delay_num: int = def_end_delay_num):
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

    def save_frames(self, beginTime, endTime, outPath, begin_last_num: int = def_begin_last_num,
                    end_delay_num: int = def_end_delay_num):
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
                return
            begin_frame_num = max(0, begin_frame_num - begin_last_num)
            end_frame_num = min(end_frame_num + end_delay_num + out_fps, lenthThis - 1)
            try:
                if _ffmpeg_use_nvenc():
                    with imageio.get_writer(
                            outPath,
                            fps=out_fps,
                            codec="h264_nvenc",
                            pixelformat="yuv420p",
                            output_params=[
                                '-cq', '30',
                                '-preset', FFMPEG_NVENC_PRESET,
                            ]
                    ) as writer:
                        for i in range(begin_frame_num, end_frame_num):
                            (frame, formatted_time) = self.buffer[i]
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            writer.append_data(rgb_frame)
                else:
                    with imageio.get_writer(
                            outPath,
                            fps=out_fps,
                            codec="libx264",
                            pixelformat="yuv420p",
                            quality=6,
                            output_params=[
                                '-crf', '30',
                                '-preset', 'fast',
                                '-movflags', '+faststart'  # Web优化：立即播放
                            ]
                    ) as writer:
                        for i in range(begin_frame_num, end_frame_num):
                            (frame, formatted_time) = self.buffer[i]
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            writer.append_data(rgb_frame)
            except Exception as e:
                print(f"保存视频时发生错误：{e}")
                time.sleep(3)
                if os.path.exists(outPath):
                    os.remove(outPath)


# endregion

# region 截断前后，保留中间帧，合并视频
class SaveMixVideos():
    def __init__(self,videos: list[str], outPath: str):
        self.videos = videos
        self.outPath = outPath
        self.frames = list()
    def MakeFrames(self):
        for i, video in enumerate(self.videos):
            cap = cv2.VideoCapture(video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameIndex:int = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                add:bool=False
                if i==0:
                    if frameIndex<total_frames-def_end_delay_num:
                        add=True
                elif i==len(self.videos)-1:
                    if frameIndex>=def_begin_last_num:
                        add=True
                else:
                    if frameIndex>=def_begin_last_num and frameIndex<total_frames-def_end_delay_num:
                        add=True
                if add:
                    self.frames.append(frame)
                frameIndex += 1
            cap.release()
        print("action_frames length:",len(self.frames))
    
    def SaveFramesAsVideo(self):
        try:
            if _ffmpeg_use_nvenc():
                with imageio.get_writer(
                        self.outPath,
                        fps=out_fps,
                        codec="h264_nvenc",
                        pixelformat="yuv420p",
                        output_params=[
                            '-cq', '30',
                            '-preset', FFMPEG_NVENC_PRESET,
                        ]
                ) as writer:
                    for i in range(len(self.frames)-1):
                        rgb_frame = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2RGB)
                        writer.append_data(rgb_frame)
            else:
                with imageio.get_writer(
                        self.outPath,
                        fps=out_fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        quality=6,
                        output_params=[
                            '-crf', '30',
                            '-preset', 'fast',
                            '-movflags', '+faststart'  # Web优化：立即播放
                        ]
                ) as writer:
                    for i in range(len(self.frames)-1):
                        rgb_frame = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2RGB)
                        writer.append_data(rgb_frame)
        except Exception as e:
            print(f"保存视频时发生错误：{e}")
            time.sleep(3)
            if os.path.exists(self.outPath):
                os.remove(self.outPath)

# endregion


# region 视频图片保存
# 在保存视频函数中，使用原始帧（未处理）
def save_frames_as_video(frames, output_path):
    # print("action_frames length:",len(frames))
    # '''
    try:
        rgb_frames = [cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB) for (frame, formatted_time) in frames]
        # macro_block_size=8,宽高设为16的整数就不需要这个参数了
        # CPU处理----由于推流服务一直运行，完全CPU处理会导致推流出现卡顿，特别是动作视频保存时候
        # with imageio.get_writer(
        #         output_path,
        #         fps=out_fps,
        #         codec="libx264",
        #         pixelformat="yuv420p",
        #         quality=6,
        #         output_params=[
        #             '-crf', '30',
        #             '-preset', 'fast',
        #             '-movflags', '+faststart' # Web优化：立即播放
        #         ]
        # ) as writer:

        # NV显卡处理---理论上推流和存视频是调用GPU不同模块，但.......
        with imageio.get_writer(
                output_path,
                fps=out_fps,
                codec="h264_nvenc",
                pixelformat="yuv420p",
                output_params=[
                    '-cq', '30',
                    '-preset', FFMPEG_NVENC_PRESET,
                ]
        ) as writer:

            # 集成显卡处理
            # with imageio.get_writer(
            #         output_path,
            #         fps=out_fps,
            #         codec="h264_vaapi",
            #         pixelformat="yuv420p",
            #         output_params=[
            #             "-vaapi_device", "/dev/dri/renderD128",
            #             "-qp", "20"
            #         ]   #老版本注意顺序，暂无驱动
            # ) as writer:
            for rgb_frame in rgb_frames:
                writer.append_data(rgb_frame)
        # CPU，下面是硬件加速
        # with imageio.get_writer(
        #         output_path,
        #         fps=org_fps,
        #         codec="h264_nvenc",
        #         pixelformat="yuv420p",  # 兼容格式
        #         ffmpeg_params=[
        #             "-vf", "scale=1920:1080:force_original_aspect_ratio=disable",  # 强制原始分辨率
        #             '-cq', '30',
        #             '-preset', 'fast',
        #             '-rc', 'vbr',       # 恒定质量模式
        #         ]
        # ) as writer:


    except Exception as e:
        print(f"保存视频时发生错误：{e}")
        time.sleep(3)
        if os.path.exists(output_path):
            os.remove(output_path)
    # """将帧序列保存为MP4视频"""
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # for frame, _ in frames:
    #     out.write(frame)  # 这里使用的是缓冲区存储的原始帧
    # out.release()
    # '''
    print(output_path)
    # return output_path


def Run_SaveVideo(stop_event: ThreadPoolEvent, frameBuffer: FrameBuffer, output_path: str, beginTime: str,
                  endTime: str):
    print("Run_SaveVideo")
    try:
        if stop_event.is_set():
            print("保存视频失败")
            return
        frameBuffer.save_frames(beginTime, endTime, output_path)
    except Exception as e:
        print(f"处理失败: {str(e)}---{output_path}")
        raise RuntimeError(f"处理失败: {str(e)}")


def save_frames_as_image(frames, output_path):
    try:
        resized_image = cv2.resize(frames, (buffer_frame_width, buffer_frame_height))
        cv2.imwrite(output_path, resized_image)
    except Exception as e:
        time.sleep(1)
        print(f"保存图片时发生错误：{e}")
        if os.path.exists(output_path):
            os.remove(output_path)


# endregion
# region 中文绘制函数
def draw_chinese(image, text, position, color):
    """使用PIL绘制中文文本"""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# endregion


# region 单贞处理函数
# YOLO单帧处理
def yolo_process_single_frame(single_frame, yolo_model, THRESH, begin_beat_name, beat_list):
    """处理单帧并进行推理"""
    # 获取帧宽度
    frame_width = single_frame.shape[1]
    print(f"当前帧宽度: {frame_width}像素")  # 调试用
    """处理单帧并进行推理"""
    # , stream=True ,half=True,
    results = yolo_model.predict(single_frame, conf=THRESH, iou=THRESH,
                                 device=YOLOV8_DEVICE)
    detections = []  # 存储检测结果的列表
    for result in results:
        # 绘制分割掩码
        if result.masks is not None:
            for mask in result.masks.xy:
                pts = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.polylines(single_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=line_size)

        # 遍历每个检测框
        for i, (box, cls, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = yolo_model.names[class_id]
            confidence = conf.item()  # 转换为Python float

            # 获取中文标签（需要自己维护映射字典）
            chinese_labels = {}
            label = chinese_labels.get(class_name, class_name)

            is_result = True
            # 标签获取节拍名
            beat_name = find_beat_name_by_label(beat_list, label)
            print(beat_name)
            print(x1)
            # 判断标签获取的节拍名不等于开始节拍或者获取的标签名不包含涂油
            # 并且x1的左边点大于屏幕宽度的一半的时候
            if (beat_name != begin_beat_name or '涂油' not in label) and x1 > frame_width // 2:
                is_result = False
            
            # if ("ANT插排线_WIFIA" == label and confidence<0.95):
            #     is_result = False

            if is_result:
                # 存储检测信息 (标签, 置信度, 坐标)
                detections.append({
                    "labelName": label,
                    "conf": confidence,
                    # "isResult": is_result,
                    # "bbox": [x1, y1, x2, y2],
                    # "class_name": class_name
                })

            # 获取颜色并绘制
            color = color_palette.get(class_name, (0, 255, 255))
            cv2.rectangle(single_frame, (x1, y1), (x2, y2), color, line_size)

            # 绘制中文标签（需实现draw_chinese函数）
            text = f"{label} {confidence:.2f}"
            single_frame = draw_chinese(single_frame, text, (x1, y1 - 30), color)

    return single_frame, detections


# endregion


# region 初始化ffmpeg推流服务
class VideoProcessor:
    def __init__(self, rtsp_outPut: str):
        self.rtspOutPut = rtsp_outPut
        self.ffmpegProcess = self._init_ffmpeg()

    # 初始化ffmpeg进程
    def _init_ffmpeg(self):
        #          '-hwaccel', 'cuda',
        #         '-max_muxing_queue_size', '1024',
        # '-gpu_memory', f'200',  # 每流200MB显存
        # command = [
        #     'ffmpeg',
        #     '-hwaccel', 'cuda',
        #     '-y',
        #     '-f', 'rawvideo',
        #     '-vcodec', 'rawvideo',
        #     '-pix_fmt', 'bgr24',
        #     '-s', f'{output_width}x{output_height}',
        #     '-r', str(out_fps),
        #     '-i', '-',
        #     '-c:v', 'libx264',
        #     '-rtsp_transport', 'tcp',
        #     '-preset', 'ultrafast',
        #     '-tune', 'zerolatency',
        #     '-f', 'rtsp',
        #     self.rtspOutPut
        # ]
        # [
        #     'ffmpeg',
        #     '-hwaccel', 'cuda',
        #     "-hwaccel_output_format", "cuda",  # 显存直接传递数据
        #     '-f', 'rawvideo',
        #     '-vcodec', 'rawvideo',
        #     '-pix_fmt', 'bgr24',
        #     '-s', f'{output_width}x{output_height}',
        #     '-r', str(out_fps),
        #     '-i', '-',
        #     # 视频编码参数
        #     '-vf', 'format=nv12',  # 或者使用 'format=yuv420p'，但nv12是硬件加速编码器更喜欢的
        #     "-c:v", "h264_nvenc",  # NVIDIA H.264编码
        #     "-preset", "ll",  # FFmpeg 4.x 用 ll/fast；p3 仅新版 FFmpeg
        #     "-b:v", "2M",  # 目标码率
        #     "-g", "12",  # 关键帧间隔
        #     "-profile:v", "main",  # H.264配置
        #     '-rtsp_transport', 'tcp',
        #     '-f', 'rtsp',
        #     self.rtspOutPut
        # ]
        # 有 CUDA 且允许时用 NVENC；否则全平台统一 CPU libx264（避免 Linux 无显卡时 ffmpeg 秒退导致 Broken pipe）。
        cpu_rtsp_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{output_width}x{output_height}",
            "-r",
            str(out_fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-g",
            "12",
            "-rtsp_transport",
            "tcp",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-f",
            "rtsp",
            self.rtspOutPut,
        ]
        nvenc_rtsp_cmd = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{output_width}x{output_height}",
            "-r",
            str(out_fps),
            "-i",
            "-",
            "-c:v",
            "h264_nvenc",
            "-preset",
            FFMPEG_NVENC_PRESET,
            "-b:v",
            "2M",
            "-g",
            "12",
            "-profile:v",
            "main",
            "-rtsp_transport",
            "tcp",
            "-f",
            "rtsp",
            self.rtspOutPut,
        ]
        command = nvenc_rtsp_cmd if _ffmpeg_use_nvenc() else cpu_rtsp_cmd
        return subprocess.Popen(command, stdin=subprocess.PIPE)

    def write_frame(self, frame, id, redis):
        try:
            self.ffmpegProcess.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"推流失败: {str(e)}")
            statusKey = REDIS_KEY_PROCESS_STATUS + str(id)
            redis.set(statusKey, "end")  # 停止推理，最多1分钟后重启
            # self.ffmpegProcess = self._init_ffmpeg(self.rtspOutPut)  # 重启推流进程

    def close(self):
        self.ffmpegProcess.stdin.close()
        self.ffmpegProcess.wait()


def runVideoStream(streamProcess: VideoProcessor, frame, rtsp_outPut, id, redis):
    if streamProcess is None:
        streamProcess = VideoProcessor(rtsp_outPut)
    streamProcess.write_frame(frame, id, redis)


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

# region  MASKRCNN模型创建函数

def create_cnn_model(num_classes, box_thresh=0.8):
    backbone = resnet50_fpn_backbone(pretrain_path=TRAIN_ORGMODEL_PATH + "/resnet50-0676ba61.pth", trainable_layers=1)
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model


def cnn_process_single_frame(single_frame, model, device, category_index, THRESH, begin_beat_name, beat_list):
    """处理单帧并进行推理"""
    # 获取帧宽度
    frame_width = single_frame.shape[1]
    print(f"当前帧宽度: {frame_width}像素")  # 调试用
    """处理单帧并进行推理"""
    # , stream=True ,half=True,
    rgb_frame = cv2.cvtColor(single_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    # pil_image = Image.open(img_path).convert("RGB")
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(pil_image)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    img_height, img_width = img.shape[-2:]
    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
    model(init_img)
    prediction = model(img.to(device))[0]
    predict_boxes = prediction["boxes"].to("cpu").detach().numpy()
    predict_classes = prediction["labels"].to("cpu").detach().numpy()
    predict_scores = prediction["scores"].to("cpu").detach().numpy()
    predict_mask_org = prediction["masks"]
    predict_mask = np.squeeze(predict_mask_org.cpu().detach().numpy(), axis=1)  # [batch, 1, h, w] -> [batch, h, w]
    # predict_mask = np.where(predict_mask > 0.7, True, False)
    # print("predict_mask:",predict_mask)
    maskss = np.where(predict_mask > THRESH, True, False)
    detections = []  # 存储检测结果的列表
    if len(predict_boxes) == 0:
        print("没有检测到任何目标!")
    else:
        print(f"检测阈值{THRESH},检测到{len(predict_boxes)}个目标!")
        segmentations = masks_to_coco_polygon(maskss)
        results = []
        for i in range(len(predict_scores)):
            # 解析检测信息
            label_id = str(predict_classes[i])
            box = predict_boxes[i].astype(int)
            score = predict_scores[i]
            label = category_index.get(label_id, 'N/A')
            # 获取坐标
            x1, y1, x2, y2 = predict_boxes[i].astype(int)
            is_result = True
            # 标签获取节拍名
            beat_name = find_beat_name_by_label(beat_list, label)
            # 判断标签获取的节拍名不等于开始节拍或者获取的标签名不包含涂油
            # 并且x1的左边点大于屏幕宽度的一半的时候
            if (beat_name != begin_beat_name or '涂油' not in label) and x1 > frame_width // 2:
                is_result = False
            if is_result:
                detections.append({
                    "labelName": label,
                    "conf": float(score),
                    # "isResult": is_result,
                    # "bbox": [x1, y1, x2, y2],
                    # "class_name": class_name
                })
                # 记录结果数据
                results.append({
                    "label": label,
                    "labelid": label_id,
                    "score": float(score),
                    "bbox": box.tolist(),
                    "segmentation": [segmentations[i]]
                })

        plot_img = draw_objs(pil_image,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             box_thresh=THRESH,
                             mask_thresh=THRESH,
                             category_index=category_index,
                             line_thickness=line_size,
                             font=font_path,
                             font_size=font_size)
        single_frame = cv2.cvtColor(np.array(plot_img), cv2.COLOR_RGB2BGR)

    return single_frame, detections


# endregion


# region java接口调用封装
def post_java_process(config):
    threading.Thread(target=postJsonWithOutJwt,
                     args=(JAVA_API_PATH + SEND_PROCESS_RESULT_TO_JAVA, config)).start()


# endregion


# 启动推理
# 测试加解密ttttt22233344555666
def start_process_video_stream(config: ProcessorConfig, redis: RedisManager):
    error_catcher = FFmpegErrorCatcher()
    # region 初始化缓存名称
    # 节拍名
    beat_name_key = REDIS_KEY_BEAT_NAME + str(config.id)
    # 节拍最早时间
    beat_time_key = REDIS_KEY_BEAT_BEGINTIME + str(config.id)
    # 节拍本地最早时间
    beat_local_time_key = REDIS_KEY_BEAT_LOCAL_BEGINTIME + str(config.id)
    # 标签名
    label_name_key = REDIS_KEY_LABEL_NAME + str(config.id)
    # 标签最早时间
    label_time_key = REDIS_KEY_LABEL_BEGINTIME + str(config.id)
    # 动作开始时间
    action_time_key = REDIS_KEY_BEGIN_BEAT_TIME + str(config.id)
    # 动作序号
    action_code_key = REDIS_KEY_ACTION_CODE + str(config.id)

    # endregion

    # region 初始化节拍列表  开始和结束节拍
    beat_list: list[BeatInfo] = config.beatList
    # 开始节拍
    begin_beat_name = beat_list[0].beatName
    # 结束节拍
    last_beat_name = beat_list[-1].beatName
    # endregion

# region 创建节拍时间限制映射
    beat_time_limits = {}
    for beat in beat_list:
        beat_time_limits[beat.beatName] = {
            'min': getattr(beat, 'beatLowerTime', 0),
            'max': getattr(beat, 'beatUpperTime', 10),
            'beatId': beat.id
        }
# endregion

    # get devices
    cnndevice = torch.device(MASKRCNN_DEFALT_DEVICE if torch.cuda.is_available() else "cpu")
    category_index = None
    # print("using {} device.".format(device))

    # 加载训练权重文件
    model = None
    if config.actionType == "1":
        num_classes = 0  # 不包含背景
        box_thresh = config.minThresh  # 检测阈值
        weights_path = DEDUCTION_MODEL_PATH + "/" + config.actionId + "_" + config.actionType + ".pth"
        label_json_path = TRAIN_DATASETS_SAVE_PATH + "/" + config.actionId + "_classes.json"
        # read class_indict
        assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
        with open(label_json_path, 'r', encoding="utf-8") as json_file:
            category_index: dict = json.load(json_file)
        num_classes = len(category_index)
        # create model
        model = create_cnn_model(num_classes=num_classes + 1, box_thresh=box_thresh)
        # load train weights
        assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
        weights_dict = torch.load(weights_path, map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        model.load_state_dict(weights_dict)
        model.to(cnndevice)
        model.eval()  # 进入验证模式
    elif config.actionType == "2":
        model = YOLO(DEDUCTION_MODEL_PATH + '/' + config.actionId + "_" + config.actionType + '.pt')
        print(model)
    else:
        print("未知动作类型")
        return
    # 加载得视频流url或者视频文件得路劲
    # video_url = 'D:/WuJianPing/company_project/gr_dk/fjy/截取工艺视频/3_output_2.mp4'
    #  获取视频帧的分辨率(高和宽)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取视频帧的帧率
    # frame_fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频保存器(将推流后得视频保存指定目录)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用'XVID'
    # out = cv2.VideoWriter(
    #     os.path.join(output_dir, uuid_str + '.mp4'),
    #     fourcc,
    #     # 10,
    #     frame_fps if frame_fps != 0 else 30,  # 如果获取不到fps则默认30
    #     (frame_width, frame_height)
    # )
    # 初始化ffmpeg推流
    ffmpegProcess = VideoProcessor(config.ffurl)
    buffer = FrameBuffer(delay_frames=def_delay_frames_num, max_frame_num=def_max_frame_num)
    statusKey = REDIS_KEY_PROCESS_STATUS + str(config.id)
    i: int = 0
    # frame_count = 0
    # 动作判定结果  1：OK 2：NG，3：未开始（默认3）
    action_status = "1"
    # 叠加动作日志格式:{节拍1}:{节拍1日志};{节拍2}:{节拍2日志}------记得是叠加
    action_logs = None
    # 初始化视频流或者视频文件
    # pipeline = (
    #     'rtspsrc location='+config.rtspurl+' latency=0 ! '
    #     'rtph264depay ! h264parse ! avdec_h264 ! '
    #     'videoconvert ! appsink drop=true sync=false'
    # )
    cap = cv2.VideoCapture(config.rtspurl, cv2.CAP_FFMPEG)
    code_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        # 检查是否有解码错误
        error = error_catcher.get_error()
        if error:
            print(f"捕获H.264解码错误: MB({error['mb_x']},{error['mb_y']})")
            print(f"错误详情: {error['message']}")

            # 错误处理策略:
            if not ret:
                # 1. 尝试跳过当前帧
                continue
            else:
                print("错误处理策略: 跳过当前帧")
                continue
                # 2. 标记损坏帧
                # cv2.putText(frame, "CORRUPTED FRAME", (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # break
        if not ret:
            # 常规错误处理
            print("帧读取失败，尝试重新连接...")
            cap.release()
            cap = cv2.VideoCapture(config.rtspurl, cv2.CAP_FFMPEG)
            continue

        # processed_frame = process_single_frame(frame, model)
        if i % 5 == 1:
            # print("处理第{}帧".format(i))
            # 处理当前帧，将原来1080P缩放0.6，节约缓冲区内容
            frame = cv2.resize(frame, (buffer_frame_width, buffer_frame_height))
            buffer.add_frame(frame)
            # '''
            if redis is not None:
                if redis.get(statusKey) == "end":
                    redis.delete(statusKey)
                    break
            else:
                print("缓存丢失，暂停")
                break

            delayed_frame = buffer.get_delay_frame()
            if delayed_frame is not None:
                frame_data, timestamp = delayed_frame

                # 创建副本进行处理
                process_frame = frame_data.copy()  # 关键修改：创建处理副本
                processed_frame, detection_list = yolo_process_single_frame(process_frame, model, config.minThresh,
                                                                            begin_beat_name, beat_list) \
                    if config.actionType == "2" else cnn_process_single_frame(process_frame, model, cnndevice,
                                                                              category_index, config.minThresh,
                                                                              begin_beat_name, beat_list)
                # 创建保存推理贞图片目录
                # 获取当前日期时间 格式化为yyyymmdd字符串
                current_datetime = datetime.now()
                formatted_date = current_datetime.strftime("%Y%m%d")
                # 保存的目标路径
                image_folder_path = FILES_PATH + DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date
                video_folder_path = FILES_PATH + DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + formatted_date
                os.makedirs(image_folder_path, exist_ok=True)
                os.makedirs(video_folder_path, exist_ok=True)

                # 构造贞图片输出文件名
                new_id = uuid.uuid4().hex
                # 将UUID转换为字符串，并去掉连字符
                frame_name = f"{new_id}.jpg"
                output_path = os.path.join(image_folder_path, frame_name)

                logs_content = None
                log_lv = None
                log_type = None
                # ng_value = None
                action_long = None
                video_url = None
                beat_begin_time = None
                action_begin_time = None

                is_java_send = True
                is_have_third = False  # 是否触发逻辑3
                # 获取当前日期 格式化为yyyymmdd字符串
                code_date = datetime.now().strftime("%Y%m%d")
                print(code_date)
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
                    # 动作序号
                    redis_action_code = redis.get(action_code_key)
                    print(redis_action_code)

                    '''重写日志逻辑'''
                    '''逻辑1直接自动判断动作开始时间不为空得时候超过100秒就自动判定ng'''
                    '''这个是动作NG没有报警和节拍日志'''
                    if redis_action_time is not None:
                        # 字符串转datetime对象
                        time1 = datetime.strptime(redis_action_time, "%Y-%m-%d %H:%M:%S")
                        time2 = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        # 计算秒数差
                        action_long = (time2 - time1).total_seconds()
                        if action_long > max_action_time:
                            is_have_third = True
                            redis.delete(action_time_key)
                            redis.delete(beat_name_key)
                            redis.delete(beat_time_key)
                            # 叠加日志
                            if action_logs is None:
                                action_logs = f'<i><span style="color:red;">严重超时NOK</span>:动作【{config.actionName}】超过{max_action_time}秒未结束</i>'
                            else:
                                action_logs = action_logs + f'<i><span style="color:red;">严重超时NOK</span>:动作【{config.actionName}】超过{max_action_time}秒未结束</i>'
                                # 保存视频调用接口
                            # action_frames = buffer.get_frames_bytime(redis_beat_time, timestamp)
                            # if action_frames:
                            # 生成唯一视频文件名
                            video_action_id = uuid.uuid4().hex
                            video_name = f"{video_action_id}.mp4"
                            video_path = os.path.join(video_folder_path, video_name)
                            if redis_action_code is None:
                                code_count = 1
                                code = f"{code_date}:{config.id}{code_count:04d}"
                                print(f"当前序号: {code}")
                                # 存储到Redis
                                redis.set(action_code_key, code)
                                # 设置序号为第一个
                                redis_action_code = code

                            # 线程保存视频
                            taskid = task_manager.start_task(Run_SaveVideo, 0, None, buffer, video_path,
                                                             redis_action_time, timestamp)
                            # threading.Thread(target=save_frames_as_video,args=(action_frames, video_path)).start()
                            # 调用接口
                            json_data = {
                                "monitorConfigId": config.monitorConfigId,
                                "actionId": config.actionId,
                                "actionName": config.actionName,
                                "actionStatus": '2',
                                "actionLogs": action_logs,
                                "actionLong": action_long,
                                "logType": '2',
                                "videoPath": DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + formatted_date + '/' + video_name,
                                "actionBeginTime": redis_action_time,
                                "endTime": timestamp,
                                "processCode": redis_action_code,
                            }
                            post_java_process(json_data)
                            action_status = '1'
                            action_logs = None
                            # 获取前面的日期字符串
                            date_part = redis_action_code.split(":")[0]
                            # 获取:后面的数值
                            date_code = int(redis_action_code.split(":")[1][-4:])
                            if code_date != date_part:
                                print("缓存日期和当前日期不相等相等")
                                code_count = 1
                                code = f"{code_date}:{config.id}{code_count:04d}"
                                print(f"当前序号: {code}")
                                # 存储到Redis
                                redis.set(action_code_key, code)
                            else:
                                code_count = date_code + 1
                                code = f"{code_date}:{config.id}{code_count:04d}"
                                redis.set(action_code_key, code)

                                # ============== 新增：节拍时长实时校验 ==============
                                '''实时检查当前节拍是否超时或过快'''
                                if redis_beat_time is not None and redis_beat_name is not None:
                                    # 计算当前节拍已持续时间
                                    time1 = datetime.strptime(redis_beat_time, "%Y-%m-%d %H:%M:%S")
                                    time2 = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                    beat_duration = (time2 - time1).total_seconds()

                                    # 获取当前节拍的时间限制
                                    if redis_beat_name in beat_time_limits:
                                        time_limit = beat_time_limits[redis_beat_name]
                                        min_duration = time_limit['min']
                                        max_duration = time_limit['max']

                                        # 检查是否超过时间限制
                                        if beat_duration > max_duration:
                                            # 节拍超时NG
                                            print(
                                                f"节拍【{redis_beat_name}】超时：{beat_duration:.1f}秒 > {max_duration}秒")
                                            log_content = f'<i><span style="color:red;">节拍超时NOK</span>:节拍【{redis_beat_name}】持续{beat_duration:.1f}秒，超过上限{max_duration}秒</i>'

                                            if action_logs is None:
                                                action_logs = log_content
                                            else:
                                                action_logs = action_logs + log_content

                                            action_status = '2'  # 标记为NG
                                            log_lv = '3'
                                            logs_content = log_content

                                            # 立即处理超时NG，保存视频并上报
                                            beat_id = time_limit['beatId']
                                            beat_begin_time = redis_beat_time

                                            # 保存视频
                                            video_beat_id = uuid.uuid4().hex
                                            video_beat_name = f"{video_beat_id}.mp4"
                                            video_beat_path = os.path.join(video_folder_path, video_beat_name)
                                            video_url = DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + formatted_date + '/' + video_beat_name

                                            taskid = task_manager.start_task(Run_SaveVideo, 0, None, buffer,
                                                                             video_beat_path, redis_beat_time,
                                                                             timestamp)

                                            # 上报节拍超时日志
                                            json_data = {
                                                "monitorConfigId": config.monitorConfigId,
                                                "actionId": config.actionId,
                                                "camerasId": config.camerasId,
                                                "beatId": beat_id,
                                                "beatName": redis_beat_name,
                                                "labels": "节拍超时",
                                                "logsContent": logs_content,
                                                "logLv": log_lv,
                                                "imagePath": DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date + '/' + frame_name,
                                                "actionTime": timestamp,
                                                "logType": "1",
                                                "videoPath": video_url,
                                                "beatBeginTime": beat_begin_time,
                                                "actionBeginTime": action_begin_time,
                                                "endTime": timestamp,
                                                "actionStatus": action_status,
                                                "actionLogs": action_logs,
                                                "processCode": redis_action_code,
                                            }
                                            post_java_process(json_data)

                                            # 清理缓存，准备重新开始
                                            redis.delete(beat_name_key)
                                            redis.delete(beat_time_key)
                                            redis.delete(action_time_key)
                                            action_status = '1'
                                            action_logs = None
                                            # 跳过后续处理，直接推流
                                            resized_frame = cv2.resize(processed_frame, (output_width, output_height))
                                            threading.Thread(target=runVideoStream,
                                                             args=(ffmpegProcess, resized_frame, config.httpurl,
                                                                   config.id, redis)).start()
                                            continue

                                        # 检查是否过快（只有在节拍切换时才检查，这里只记录时长）
                                        # 实际的时间过短检查将在节拍切换时进行（见下方修改）
                                # ============== 新增结束 ==============

                    if is_have_third == False:
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
                                    if beat_long > 20:
                                        redis.delete(beat_name_key)
                                        redis.delete(beat_time_key)
                                        redis.delete(action_time_key)
                                        action_status = '1'
                                        action_logs = None
                        else:
                            '''业务处理获取推理得到得标签名和匹配率'''
                            '''增加判定如果获取多个标签判断是否是同一个节拍不是动作NG'''
                            result_map = {}
                            for obj in detection_list:
                                # 可选：添加对象有效性检查
                                if not obj or 'labelName' not in obj or 'conf' not in obj:
                                    print("遇到无效检测结果，跳过")
                                    continue
                                key = obj['labelName']
                                value = f"{obj['conf']:.2f}"
                                # isResult = true 才进行处理
                                # if obj['isResult']:
                                #     key = obj['labelName']
                                #     value = f"{obj['conf']:.2f}"
                                # 如果 key 已存在，则保留最大值；否则直接插入
                                if key in result_map:
                                    if value > result_map[key]:
                                        result_map[key] = value
                                else:
                                    result_map[key] = value

                            # 判断result_map是否为空，为空则跳过后续处理
                            # if not result_map:
                            #     continue
                            # 获取所有标签对应的节拍名称并检查一致性
                            beat_names = set()
                            for label in result_map:
                                beat_name = find_beat_name_by_label(beat_list, label)
                                beat_names.add(beat_name)
                            # 检查标签获取得所有节拍名称是否一致
                            if len(beat_names) != 1:
                                print(f"警告：检测到多个不同的节拍名称 {beat_names}")
                                # 记录日志
                                action_status = '2'
                                # 叠加日志
                                if action_logs is None:
                                    action_logs = f'<i><span style="color:red;">检测到多个节拍名称NOK</span>:【{beat_names}】</i>'
                                    logs_content = action_logs
                                    log_lv = '2'
                                else:
                                    action_logs = action_logs + f'<i><span style="color:red;">检测到多个节拍名称NOK</span>:【{beat_names}】</i>'
                                    logs_content = action_logs
                                    log_lv = '2'

                            # 获取值最大的 key
                            max_label_key = max(result_map, key=result_map.get)
                            # 用key获取值
                            max_label_value = result_map[max_label_key]
                            # 根据标签获取节拍名
                            list_beat_name = find_beat_name_by_label(beat_list, max_label_key)
                            # 用标签名获取节拍id
                            beat_id = next(
                                (beat.id for beat in beat_list if beat.beatName == list_beat_name), None
                            )

                            '''判断redis不是None得时候业务处理'''
                            if redis is not None:
                                # # 原来的逻辑该缓存序号只有在什么都没有且是开始节拍的时候才执行
                                # # 现在只要节拍不为空就做这个缓存的创建或者修改
                                # if redis_action_code is None:
                                #     print("没有序号缓存,创建")
                                #     # 创建序号缓存时间
                                #     # 生成带序号的代码
                                #     code = f"{code_date}:{config.id}{code_count:04d}"
                                #     print(f"当前序号: {code}")
                                #     # 存储到Redis
                                #     redis.set(action_code_key, code)
                                # else:
                                #     print("序号缓存已存在")
                                #     # 获取前面的日期字符串
                                #     date_part = redis_action_code.split(":")[0]
                                #     if code_date != date_part:
                                #         print("缓存日期和当前日期不相等相等")
                                #         code_count = 1
                                #         code = f"{code_date}:{config.id}{code_count:04d}"
                                #         print(f"当前序号: {code}")
                                #         # 存储到Redis
                                #         redis.set(action_code_key, code)
                                '''当缓存节拍和缓存节拍时间都为空得时候就默认是第一次进入'''
                                if redis_beat_time is None and redis_beat_name is None:
                                    # 原来的逻辑该缓存序号只有在什么都没有且是开始节拍的时候才执行
                                    # 现在只要节拍不为空就做这个缓存的创建或者修改
                                    if redis_action_code is None:
                                        print("没有序号缓存,创建")
                                        # 创建序号缓存时间
                                        # 生成带序号的代码
                                        code = f"{code_date}:{config.id}{code_count:04d}"
                                        print(f"当前序号: {code}")
                                        # 存储到Redis
                                        redis.set(action_code_key, code)
                                    else:
                                        print("序号缓存已存在")
                                        # 获取前面的日期字符串
                                        date_part = redis_action_code.split(":")[0]
                                        # 获取:后面的数值
                                        date_code = int(redis_action_code.split(":")[1][-4:])
                                        if code_date != date_part:
                                            print("缓存日期和当前日期不相等相等")
                                            code_count = 1
                                            code = f"{code_date}:{config.id}{code_count:04d}"
                                            print(f"当前序号: {code}")
                                            # 存储到Redis
                                            redis.set(action_code_key, code)
                                    if redis_action_time is None:
                                        redis.set(action_time_key, timestamp)
                                        redis_action_time = timestamp
                                        action_begin_time = timestamp
                                    '''当前标签获取得节拍是开始节拍判定'''
                                    '''当前获取得标签不是开始节拍判定'''
                                    if list_beat_name == begin_beat_name:
                                        print('是开始节拍')
                                        # redis.set(action_time_key, timestamp)
                                        # 记录日志 【】
                                        if log_lv != '2':
                                            logs_content = f'<i><span style="color:green;">OK</span>当前节拍:【{list_beat_name}】,标签名称: 【{max_label_key}】,匹配率:【{max_label_value}】</i>'
                                            log_lv = '1'
                                        action_status = '1'
                                    else:
                                        print("不是开始节拍")
                                        # 记录日志 【】
                                        logs_content = f'<i><span style="color:red;">无法找到起始动作NOK</span>:【{list_beat_name}】不是开始节拍无法找到起始动作,判定NG</i>'
                                        log_lv = '3'
                                        action_status = '2'
                                    # 更新缓存节拍名称和节拍开始时间
                                    redis.set(beat_name_key, list_beat_name)
                                    redis.set(beat_time_key, timestamp)
                                    # 叠加日志
                                    if action_logs is None:
                                        action_logs = f'{logs_content}'
                                    else:
                                        action_logs = action_logs + f'{logs_content}'
                                    '''第一次进入得话直接调用java接口进行新增节拍和报警日志'''
                                    json_data = {
                                        "monitorConfigId": config.monitorConfigId,
                                        "actionId": config.actionId,
                                        "camerasId": config.camerasId,
                                        "beatId": beat_id,
                                        "beatName": list_beat_name,
                                        "labels": max_label_key,
                                        "logsContent": logs_content,
                                        "logLv": log_lv,
                                        "imagePath": DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date + '/' + frame_name,
                                        "actionTime": timestamp,
                                        "logType": "1",
                                        "videoPath": video_url,
                                        "beatBeginTime": beat_begin_time,
                                        "actionBeginTime": action_begin_time,
                                        "endTime": timestamp,
                                        "actionStatus": action_status,
                                        "actionLogs": action_logs,
                                        "processCode": redis_action_code,
                                    }
                                    post_java_process(json_data)
                                    # 保存帧为图片
                                    threading.Thread(target=save_frames_as_image,
                                                     args=(processed_frame, output_path)).start()
                                else:
                                    '''当缓存节拍和节拍时间都不为空得时候且当前节拍和缓存节拍不一致'''
                                    '''两种情况一种是结束节拍一种是下一个节拍'''
                                    if list_beat_name != redis_beat_name:

                                        # ============== 新增：节拍切换时检查上一个节拍时长 ==============
                                        '''在切换节拍前，检查上一个节拍的时长是否合规'''
                                        if redis_beat_time is not None and redis_beat_name in beat_time_limits:
                                            # 计算上一个节拍的持续时间
                                            prev_beat_begin = datetime.strptime(redis_beat_time, "%Y-%m-%d %H:%M:%S")
                                            current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                            prev_beat_duration = (current_time - prev_beat_begin).total_seconds()

                                            prev_time_limit = beat_time_limits[redis_beat_name]
                                            prev_min_duration = prev_time_limit['min']
                                            prev_max_duration = prev_time_limit['max']
                                            prev_beat_id = prev_time_limit['beatId']

                                            # 检查上一个节拍是否过短
                                            if prev_beat_duration < prev_min_duration:
                                                print(
                                                    f"节拍【{redis_beat_name}】过快：{prev_beat_duration:.1f}秒 < {prev_min_duration}秒")
                                                short_log_content = f'<i><span style="color:red;">节拍过快NOK</span>:节拍【{redis_beat_name}】仅持续{prev_beat_duration:.1f}秒，低于下限{prev_min_duration}秒</i>'

                                                if action_logs is None:
                                                    action_logs = short_log_content
                                                else:
                                                    action_logs = action_logs + short_log_content

                                                action_status = '2'  # 标记为NG
                                                log_lv = '3'
                                                logs_content = short_log_content

                                                # 保存过快节拍的视频
                                                video_short_id = uuid.uuid4().hex
                                                video_short_name = f"{video_short_id}.mp4"
                                                video_short_path = os.path.join(video_folder_path, video_short_name)
                                                video_short_url = DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + formatted_date + '/' + video_short_name

                                                taskid = task_manager.start_task(Run_SaveVideo, 0, None, buffer,
                                                                                 video_short_path, redis_beat_time,
                                                                                 timestamp)

                                                # 上报节拍过快日志
                                                short_json_data = {
                                                    "monitorConfigId": config.monitorConfigId,
                                                    "actionId": config.actionId,
                                                    "camerasId": config.camerasId,
                                                    "beatId": prev_beat_id,
                                                    "beatName": redis_beat_name,
                                                    "labels": "节拍过快",
                                                    "logsContent": logs_content,
                                                    "logLv": log_lv,
                                                    "imagePath": DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date + '/' + frame_name,
                                                    "actionTime": timestamp,
                                                    "logType": "1",
                                                    "videoPath": video_short_url,
                                                    "beatBeginTime": redis_beat_time,
                                                    "actionBeginTime": action_begin_time,
                                                    "endTime": timestamp,
                                                    "actionStatus": action_status,
                                                    "actionLogs": action_logs,
                                                    "processCode": redis_action_code,
                                                }
                                                post_java_process(short_json_data)

                                                # 清理缓存，准备重新开始
                                                redis.delete(beat_name_key)
                                                redis.delete(beat_time_key)
                                                redis.delete(action_time_key)
                                                action_status = '1'
                                                action_logs = None
                                                # 跳过后续节拍切换处理，直接推流
                                                resized_frame = cv2.resize(processed_frame,
                                                                           (output_width, output_height))
                                                threading.Thread(target=runVideoStream,
                                                                 args=(ffmpegProcess, resized_frame, config.httpurl,
                                                                       config.id, redis)).start()
                                                continue
                                        # ============== 新增结束 ==============

                                        # 原来的逻辑该缓存序号只有在什么都没有且是开始节拍的时候才执行
                                        # 现在只要节拍不为空就做这个缓存的创建或者修改
                                        if redis_action_code is None:
                                            print("没有序号缓存,创建")
                                            # 创建序号缓存时间
                                            # 生成带序号的代码
                                            code = f"{code_date}:{config.id}{code_count:04d}"
                                            print(f"当前序号: {code}")
                                            # 存储到Redis
                                            redis.set(action_code_key, code)
                                        else:
                                            print("序号缓存已存在")
                                            # 获取前面的日期字符串
                                            date_part = redis_action_code.split(":")[0]
                                            # 获取:后面的数值
                                            date_code = int(redis_action_code.split(":")[1][-4:])
                                            if code_date != date_part:
                                                print("缓存日期和当前日期不相等相等")
                                                code_count = 1
                                                code = f"{code_date}:{config.id}{code_count:04d}"
                                                print(f"当前序号: {code}")
                                                # 存储到Redis
                                                redis.set(action_code_key, code)
                                        if redis_action_time is None:
                                            redis.set(action_time_key, timestamp)
                                            redis_action_time = timestamp
                                            action_begin_time = timestamp
                                        '''缓存节拍是结束节拍'''
                                        '''两种情况一种是当前节拍是开始节拍一种是其他节拍'''
                                        if list_beat_name == begin_beat_name:
                                            # 当前节拍是开始节拍
                                            # redis.set(action_time_key, timestamp)
                                            # action_begin_time = timestamp
                                            # 记录日志 【】
                                            if log_lv != '2':
                                                logs_content = f'<i><span style="color:green;">OK</span>:当前节拍: 【{list_beat_name}】,标签名称: 【{max_label_key}】,匹配率:【{max_label_value}】</i>'
                                                log_lv = '1'
                                            # 该缓存序号只有在什么都没有且是开始节拍的时候才执行
                                            # if redis_action_code is None:
                                            #     print("没有序号缓存,创建")
                                            #     # 创建序号缓存时间
                                            #     # 生成带序号的代码
                                            #     code = f"{code_date}:{config.id}{code_count:04d}"
                                            #     print(f"当前序号: {code}")
                                            #     # 存储到Redis
                                            #     redis.set(action_code_key, code)
                                            # else:
                                            #     print("序号缓存已存在")
                                            #     # 获取前面的日期字符串
                                            #     date_part = redis_action_code.split(":")[0]
                                            #     if code_date != date_part:
                                            #         print("缓存日期和当前日期不相等相等")
                                            #         code_count = 1
                                            #         code = f"{code_date}:{config.id}{code_count:04d}"
                                            #         print(f"当前序号: {code}")
                                            #         # 存储到Redis
                                            #         redis.set(action_code_key, code)
                                        else:
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
                                                # 记录日志 【】
                                                # <i><span style="color:red;">严重超时NOK</span>:动作{config.actionName}超过{max_action_time}秒未结束</i>
                                                if log_lv != '2':
                                                    logs_content = f'<i><span style="color:green;">OK</span>:当前节拍: 【{list_beat_name}】,标签名称: 【{max_label_key}】,匹配率:【{max_label_value}】</i>'
                                                    log_lv = '1'
                                            else:
                                                # 记录日志
                                                logs_content = f'<i><span style="color:red;">工序混乱NOK</span>:当前节拍:【{list_beat_name}】,不是节拍:【{redis_beat_name}】的下一个节拍</i>'
                                                log_lv = '3'
                                                action_status = '2'

                                        '''缓存节拍和当前节拍不一致得情况下要记录报警日志和节拍日志'''
                                        '''记录节拍日志得时候还需要保存视频'''
                                        # region
                                        # if redis_beat_name == last_beat_name:
                                        #     '''缓存节拍是结束节拍'''
                                        #     '''两种情况一种是当前节拍是开始节拍一种是其他节拍'''
                                        #     if list_beat_name == begin_beat_name:
                                        #         # 当前节拍是开始节拍
                                        #         redis.set(action_time_key, timestamp)
                                        #         action_begin_time = timestamp
                                        #         # 记录日志
                                        #         logs_content = f'当前节拍: {list_beat_name},标签名称: {max_label_key},匹配率:{max_label_value}'
                                        #         log_lv = '1'
                                        #     else:
                                        #         # 记录日志
                                        #         logs_content = f'未捕捉起始节拍: {begin_beat_name},直接执行了: {list_beat_name}节拍,工序混乱NG'
                                        #         log_lv = '3'
                                        #         action_status = '2'
                                        # else:
                                        #     '''缓存节拍不是结束节拍'''
                                        #     '''判断当前节拍是不是缓存节拍得下一个节拍'''
                                        #     # 初始化结果
                                        #     is_next = False
                                        #     # 遍历列表（从第一个元素到倒数第二个元素）
                                        #     for i in range(len(beat_list) - 1):
                                        #         # 检查当前节拍是否为缓存的下一个节拍
                                        #         if beat_list[i].beatName == redis_beat_name and \
                                        #                 beat_list[i + 1].beatName == list_beat_name:
                                        #             is_next = True
                                        #             break
                                        #     if is_next:
                                        #         # 记录日志
                                        #         logs_content = f'当前节拍: {list_beat_name},标签名称: {max_label_key},匹配率:{max_label_value}'
                                        #         log_lv = '1'
                                        #     else:
                                        #         # 记录日志
                                        #         logs_content = f'当前节拍:{list_beat_name},不是节拍:{redis_beat_name}的下一个节拍,工序混乱NG'
                                        #         log_lv = '3'
                                        #         action_status = '2'
                                        # endregion
                                        '''上面是对节拍得判定这个是每次进来因为节拍不一致需要保存节拍日志和节拍视频'''
                                        beat_begin_time = redis_beat_time
                                        # 保存节拍视频调用接口
                                        # 线程保存视频

                                        # beat_frames = buffer.get_frames_bytime(redis_beat_time, timestamp)
                                        # if beat_frames:
                                        # 生成唯一视频文件名
                                        # 构造贞图片输出文件名
                                        video_beat_id = uuid.uuid4().hex
                                        # 将UUID转换为字符串，并去掉连字符
                                        video_beat_name = f"{video_beat_id}.mp4"
                                        video_beat_path = os.path.join(video_folder_path, video_beat_name)
                                        video_url = DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + formatted_date + '/' + video_beat_name
                                        # 保存视频
                                        # threading.Thread(target=save_frames_as_video,args=(beat_frames, video_beat_path)).start()
                                        taskid = task_manager.start_task(Run_SaveVideo, 0, None, buffer,
                                                                         video_beat_path, redis_beat_time, timestamp)
                                        json_data = {
                                            "monitorConfigId": config.monitorConfigId,
                                            "actionId": config.actionId,
                                            "camerasId": config.camerasId,
                                            "beatId": beat_id,
                                            "beatName": list_beat_name,
                                            "labels": max_label_key,
                                            "logsContent": logs_content,
                                            "logLv": log_lv,
                                            "imagePath": DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date + '/' + frame_name,
                                            "actionTime": timestamp,
                                            "logType": "1",
                                            "videoPath": video_url,
                                            "beatBeginTime": beat_begin_time,
                                            "actionBeginTime": action_begin_time,
                                            "endTime": timestamp,
                                            "actionStatus": action_status,
                                            "actionLogs": action_logs,
                                            "processCode": redis_action_code,
                                        }
                                        post_java_process(json_data)
                                        # 保存帧为图片
                                        threading.Thread(target=save_frames_as_image,
                                                         args=(processed_frame, output_path)).start()
                                        '''只要节拍更换缓存得节拍和节拍时间必定更换'''
                                        redis.set(beat_name_key, list_beat_name)
                                        redis.set(beat_time_key, timestamp)
                                        # 叠加日志
                                        if action_logs is None:
                                            action_logs = f'{logs_content}'
                                        else:
                                            action_logs = action_logs + f'{logs_content}'
                            '''最后创建'''
                            # 动作开始时间不为空 + 当前节拍为最后节拍 == == == >
                            # 计算动作时间 + 清空缓存动作开始时间 + 重置上面两个变量 == == == = > 最后统一调接口
                            if redis_action_time is not None and list_beat_name == last_beat_name:
                                print(f'{action_status}')
                                # 字符串转datetime对象
                                action_begin_time = redis_action_time
                                time1 = datetime.strptime(redis_action_time, "%Y-%m-%d %H:%M:%S")
                                time2 = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                # 计算秒数差
                                action_long = (time2 - time1).total_seconds()
                                redis.delete(action_time_key)
                                # 保存节拍视频调用接口
                                # 保存节拍视频调用接口
                                # end_video_frames = buffer.get_frames_bytime(redis_action_time,timestamp)
                                # if end_video_frames:
                                # 生成唯一视频文件名
                                # 构造贞图片输出文件名
                                end_video_beat_id = uuid.uuid4().hex
                                # 将UUID转换为字符串，并去掉连字符
                                end_video_name = f"{end_video_beat_id}.mp4"
                                end_video_path = os.path.join(video_folder_path, end_video_name)
                                end_video_url = DEDUCTION_PROCESS_LOG_VIDEO_PATH + '/' + formatted_date + '/' + end_video_name
                                # 保存视频
                                # threading.Thread(target=save_frames_as_video,args=(end_video_frames, end_video_path)).start()
                                taskid = task_manager.start_task(Run_SaveVideo, 0, None, buffer, end_video_path,
                                                                 redis_action_time, timestamp)
                                # 调接口前判断序号
                                if redis_action_code is None:
                                    code_count = 1
                                    code = f"{code_date}:{config.id}{code_count:04d}"
                                    print(f"当前序号: {code}")
                                    # 存储到Redis
                                    redis.set(action_code_key, code)
                                    # 设置序号为第一个
                                    redis_action_code = code
                                # 调用接口
                                json_data = {
                                    "monitorConfigId": config.monitorConfigId,
                                    "actionId": config.actionId,
                                    "camerasId": config.camerasId,
                                    "beatId": beat_id,
                                    "beatName": list_beat_name,
                                    "labels": max_label_key,
                                    "actionTime": timestamp,
                                    "logType": '2',
                                    "videoPath": end_video_url,
                                    "beatBeginTime": beat_begin_time,
                                    "actionBeginTime": action_begin_time,
                                    "endTime": timestamp,
                                    "actionStatus": action_status,
                                    "actionLogs": action_logs,
                                    "processCode": redis_action_code,
                                }
                                post_java_process(json_data)
                                action_status = '1'
                                action_logs = None
                                # 获取前面的日期字符串
                                date_part = redis_action_code.split(":")[0]
                                # 获取:后面的数值
                                date_code = int(redis_action_code.split(":")[1][-4:])
                                if code_date != date_part:
                                    print("缓存日期和当前日期不相等相等")
                                    code_count = 1
                                    code = f"{code_date}:{config.id}{code_count:04d}"
                                    print(f"当前序号: {code}")
                                    # 存储到Redis
                                    redis.set(action_code_key, code)
                                else:
                                    code_count = date_code + 1
                                    code = f"{code_date}:{config.id}{code_count:04d}"
                                    redis.set(action_code_key, code)

                        # 调整分辨率并推流
                        resized_frame = cv2.resize(processed_frame, (output_width, output_height))
                        threading.Thread(target=runVideoStream,
                                         args=(ffmpegProcess, resized_frame, config.httpurl, config.id, redis)).start()
        i = i + 1
        if i == 5:
            i = 0
    # 释放资源（异常保护）
    try:
        cap.release()
    except Exception:
        pass
    try:
        ffmpegProcess.close()
    except Exception:
        pass
    try:
        error_catcher.close()
    except Exception:
        pass
    # out.release()
    # cv2.destroyAllWindows()
