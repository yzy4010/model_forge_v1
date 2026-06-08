import threading, time, uuid, os, subprocess, cv2,json
from collections import deque
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from common.ajaxResult import postJsonWithOutJwt
from common.config import *
from common.redisManager import RedisManager
from datetime import datetime

from rcn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from rcn.draw_box_utils import draw_objs
from rcn.network_files.mask_rcnn import MaskRCNN
from torchvision import transforms

# 将UUID转换为字符串，并去掉连字符
uuid_str = uuid.uuid4().hex

# 创建视频保存输出目录
video_output_dir = DEDUCTION_FM_OUTPUT_VIDEO_PATH
font_path = DEDUCTION_FILE_PATH + "/simhei.ttf"  # 需要下载中文字体文件
# os.makedirs(output_dir, exist_ok=True)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # polygons = []
    #选择面积最大的轮廓
    main_contour = max(contours, key=cv2.contourArea)
    # tolerance: 简化程度(值越大点越少)
    # tolerance = 0.5
    # epsilon = tolerance * cv2.arcLength(main_contour, True)
    # main_contour = cv2.approxPolyDP(main_contour, epsilon, True)

    # for contour in contours:
    #     if len(contour) >= 3:  # 至少三个点才能形成多边形
    #         polygons.append(contour.flatten().tolist())
    polygon = main_contour.flatten().tolist()
    # print(f"polygon-----{polygon}")
    return polygon

def masks_to_coco_polygon(masks):
    """
    将多个二值掩码转换为 COCO 的 Polygon 格式 segmentation 数据
    :param masks: shape [N, H, W], bool 类型的掩码数组
    :return: list of list, 每个元素是坐标列表组成的多边形
    """
    segmentations = []
    for mask in masks:
        polygon = mask_to_polygon(mask)
        segmentations.append(polygon)
    # print(f"segmentations-----{segmentations}")
    return segmentations


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
            self.buffer.append((frame, formatted_time))

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

    def get_frames(self, last_frame_num: int = 100, begin_frame_num: int = 0):
        """获取特定帧"""
        with self.lock:
            lenthThis: int = len(self.buffer)
            if last_frame_num <= lenthThis:
                lengthOut: int = last_frame_num - begin_frame_num
                buffer = deque(maxlen=lengthOut)
                for i in lengthOut:
                    buffer.append(self.buffer[lenthThis - begin_frame_num + i])
                return buffer
            else:
                return None


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone(pretrain_path=TRAIN_ORGMODEL_PATH+"/resnet50-0676ba61.pth", trainable_layers=1)
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model




# 单贞处理
def cnn_process_single_frame(single_frame, model,device,category_index):
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
    t_start = time_synchronized()
    prediction = model(img.to(device))[0]
    t_end = time_synchronized()
    print("inference+NMS time: {}".format(t_end - t_start))
    predict_boxes = prediction["boxes"].to("cpu").numpy()
    predict_classes = prediction["labels"].to("cpu").numpy()
    predict_scores = prediction["scores"].to("cpu").numpy()
    predict_mask_org = prediction["masks"]
    predict_mask = np.squeeze(predict_mask_org.cpu().numpy(), axis=1)  # [batch, 1, h, w] -> [batch, h, w]
    # predict_mask = np.where(predict_mask > 0.7, True, False)
    # print("predict_mask:",predict_mask)
    maskss = np.where(predict_mask > MASKRCNN_DEFALT_THRESH, True, False)
    detections = []  # 存储检测结果的列表
    if len(predict_boxes) == 0:
        print("没有检测到任何目标!")
    else:
        print(f"检测阈值{MASKRCNN_DEFALT_THRESH},检测到{len(predict_boxes)}个目标!")
        segmentations=masks_to_coco_polygon(maskss)
        results = []
        for i in range(len(predict_scores)):
            # 解析检测信息
            label_id = str(predict_classes[i])
            box = predict_boxes[i].astype(int)
            score = predict_scores[i]
            label = category_index.get(label_id, 'N/A')
            detections.append({
                "labelName": label,
                "conf": float(score),
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
                            box_thresh=MASKRCNN_DEFALT_THRESH,
                            mask_thresh=MASKRCNN_DEFALT_THRESH,
                            category_index=category_index,
                            line_thickness=5,
                            font=font_path,
                            font_size=30)
        single_frame = cv2.cvtColor(np.array(plot_img), cv2.COLOR_RGB2BGR)

    return single_frame, detections


# endregion


# region 初始化ffmpeg推流服务
# 推流地址
# rtsp_output = "rtsp://localhost:8554/video_stream"
# 推流视频分辨率
output_width = 640
output_height = 360
# 推流视频贞率
fps = 10


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
    # command = [
    #     'ffmpeg',
    #     '-y',
    #     '-f', 'rawvideo',
    #     '-vcodec', 'rawvideo',
    #     '-pix_fmt', 'bgr24',
    #     '-s', f'{output_width}x{output_height}',
    #     '-r', str(fps),
    #     '-i', '-',
    #     '-c:v', 'libx264',
    #     '-preset', 'ultrafast',
    #     '-tune', 'zerolatency',
    #     +   '-x264-params', 'keyint=15:min-keyint=15',  # 减少关键帧间隔
    #     +   '-g', '15',  # GOP大小
    #     +   '-bf', '0',  # 禁用B帧
    #     +   '-profile:v', 'baseline',  # 简化profile
    #     '-f', 'rtsp',
    #     rtsp_outPut
    # ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


# endregion


# 启动推理
# mode_name 模型名称
# stream_url 视频流地址
# rtsp_output 推流地址
# def start_process_video_stream():
def startr_cnn_process_video_stream(action_id: str, stream_url: str, rtsp_outPut: str, redis: RedisManager):
    num_classes = 0  # 不包含背景
    box_thresh = MASKRCNN_DEFALT_THRESH # 检测阈值
    weights_path = DEDUCTION_MODEL_PATH+"/"+action_id+"_1.pth"
    label_json_path = TRAIN_DATASETS_SAVE_PATH+"/"+action_id+"_classes.json"
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

    # 初始化视频流或者视频文件
    cap = cv2.VideoCapture(stream_url)
    #  获取视频帧的分辨率(高和宽)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取视频帧的帧率
    # frame_fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频保存器(将推流后得视频保存指定目录)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用'XVID'
    # out = cv2.VideoWriter(
    #     os.path.join(output_dir, uuid_str + '.mp4'),
    #     fourcc,
    #     # 10,
    #     frame_fps if frame_fps != 0 else 30,  # 如果获取不到fps则默认30
    #     (frame_width, frame_height)
    # )

    # 创建保存推理贞图片目录
    # 获取当前日期时间
    current_datetime = datetime.now()
    # 格式化为yyyymmdd字符串
    formatted_date = current_datetime.strftime("%Y%m%d")
    # print(formatted_date)
    folder_path = FILES_PATH + DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date  # 替换为你的目标路径
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 自动创建多级目录
        print(f"文件夹已创建: {folder_path}")
    else:
        print(f"文件夹已存在: {folder_path}")

    # 初始化ffmpeg推流
    ffmpeg_process = init_ffmpeg(rtsp_outPut)
    buffer = FrameBuffer(delay_frames=30, max_frame_num=1200)
    i: int = 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 处理当前帧
        # processed_frame = process_single_frame(frame, model)
        if i % 3 == 1:
            buffer.add_frame(frame)
            delayed_frame = buffer.get_delay_frame()
            if delayed_frame is not None:
                frame_data, timestamp = delayed_frame
                processed_frame, detection_list = cnn_process_single_frame(frame_data, model,device,category_index)
                # 构造输出文件名 (frame_0001.jpg, frame_0002.jpg, ...)
                new_id = uuid.uuid4()
                # 将UUID转换为字符串，并去掉连字符
                frame_name = f"{str(new_id).replace('-', '')}.jpg"
                # output_path = os.path.join(folder_path, frame_name)

                # 遍历检测结果 获取标签名称和匹配率
                # 回传数据到java端处理
                # 确保detection_list是列表类型（防止返回None）
                if detection_list is None or len(detection_list) == 0:
                    print("未检测到任何对象")
                else:
                    # 调用java接口回传推理结果
                    isSend = True
                    if redis is not None:
                        labelKey = f"{REDIS_KEY_LABEL_NAME}{action_id}"
                        labelName = redis.get(labelKey)
                        if len(detection_list) == 1:
                            if labelName == detection_list[0].get("labelName"):
                                isSend = False
                        redis.set(labelKey, labelName)
                    # else:
                    #     print("未配置redis")
                    if isSend:
                        threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_PROCESS_RESULT_TO_JAVA, {
                            "actionId": action_id,
                            "actionTime": timestamp,
                            "pythonProcessLabelVoList": detection_list,
                            "imagePath": DEDUCTION_PROCESS_LOG_IMAGE_PATH + '/' + formatted_date + '/' + frame_name
                        })).start()
                        # 保存帧为图片
                        resized_image = cv2.resize(processed_frame, (960, 540))
                        # cv2.imwrite(output_path, resized_image)
                    # for obj in detection_list:
                    #     # 可选：添加对象有效性检查
                    #     if not obj or 'label' not in obj or 'confidence' not in obj:
                    #         print("遇到无效检测结果，跳过")
                    #         # continue
                    #     print(f"标签: {obj['label']}, 置信度: {obj['confidence']:.2f}")
                    #     print(f"位置: {obj['bbox']}, 原始类别: {obj['class_name']}")
                    #     print("推理结束")

                # 调整分辨率并推流
                resized_frame = cv2.resize(processed_frame, (output_width, output_height))
                # resized_frame = cv2.resize(frame, (output_width, output_height))
                try:
                    # 推流到ffmpeg
                    ffmpeg_process.stdin.write(resized_frame.tobytes())
                except Exception as e:
                    print(f"推流失败: {str(e)}")
                    ffmpeg_process = init_ffmpeg(rtsp_outPut)  # 重启推流进程

                # 显示实时画面
                # cv2.imshow('YOLOv8 Real-time Detection', processed_frame)

                # 保存处理后的帧
                # out.write(processed_frame)

                # # 退出条件
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        else:
            print("drop frame")
            # ffmpeg_process = init_ffmpeg(rtsp_outPut)  # 重启推流进程
            # resized_frame = cv2.resize(frame, (output_width, output_height))
        i = i + 1
        if i == 3:
            i = 0

    # 释放资源
    cap.release()
    if ffmpeg_process.poll() is None:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    startr_cnn_process_video_stream()
