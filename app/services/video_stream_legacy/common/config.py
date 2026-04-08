# 部署类型
PUBLIC_TYPE = "2"  # 1推理，2训练，训练部署不启动推理服务
# 推理相关配置
FILES_PATH = "/data"  # AI文件调用总路径
DEDUCTION_MODEL_PATH = "/data/train/model/actionTrue"  # 推理模型文件存放路径(如果某次训练记录被设为启用模型，则需要将模型文件或其它附属信息复制到当前目录)
DEDUCTION_ORG_VIDEO_PATH = "/data/deduction/data/videos"  # 推理时摄像头原始视频文件存放路径
DEDUCTION_ORG_VIDEO_SAVE = False  # 推理时是否保存摄像头原始视频---可能不实现这个逻辑
DEDUCTION_ORG_VIDEO_TIME_LONG: int = 180  # 推理时摄像头原始视频文件单个最长时长，秒
DEDUCTION_VIDEO_FPS: int = 30  # 推理时摄像头原始视频帧率
DEDUCTION_VIDEO_CUTE_FPS: int = 6  # 推理时摄像头截取视频帧率
DEDUCTION_VIDEO_WIDTH: int = 1920  # 推理时摄像头原始视频宽度
DEDUCTION_VIDEO_HEIGHT: int = 1080  # 推理时摄像头截取视频高度
DEDUCTION_LICENSE_NAME = "FORVIA"  # 推理时图片版权名称
DEDUCTION_FILE_PATH = "/data/deduction/data"  # 推理时文件调用路径
DEDUCTION_FM_OUTPUT_IMAGE_PATH = "/data/deduction/data/images"  # 推理时帧截图文件存放路径--用于标记临时使用，保存标签后批量复制到#TRAIN_IMAGES_SAVE_PATH#。并同步到训练服务器
DEDUCTION_FM_OUTPUT_VIDEO_PATH = "/data/deduction/data/videos"  # 推理时帧视频文件存放路径--用于标记，保存标签后批量复制到#TRAIN_VIDEOS_SAVE_PATH#。并同步到训练服务器
TRAIN_DATASETS_SAVE_PATH_AT_DEDUCTION = "/data/train/data"  # Java端[也部署在推理服务器上]完成动作和工序编排后，或动作指定了新的启用模型，将信息汇总分别生成cocojson、txt、csv格式，并将这些文件推送到训练服务器
START_VIDEO_PRE_MARK_PX = "aicache:startVideoPreMarking:id_"
START_TRAINING_PX = "aicache:startTraining:id_"
# region 缓存配置
# 节拍名
REDIS_KEY_BEAT_NAME = "aicache:beatname:id_"
# 节拍最早时间
REDIS_KEY_BEAT_BEGINTIME = "aicache:beattime:id_"
# 节拍本地最早时间
REDIS_KEY_BEAT_LOCAL_BEGINTIME = "aicache:localbeattime:id_"
# 标签名
REDIS_KEY_LABEL_NAME = "aicache:labelname:id_"
# 标签最早时间
REDIS_KEY_LABEL_BEGINTIME = "aicache:labeltime:id_"
# 动作开始时间
REDIS_KEY_BEGIN_BEAT_TIME = "aicache:begintime:id_"
# 推理是否结束
REDIS_KEY_PROCESS_STATUS = "aicache:processStatus:id_"
# 推理配置
REDIS_KEY_PROCESS_CONFIG = "aicache:processConfig:id_"
# 动作序号
REDIS_KEY_ACTION_CODE = "aicache:actioncode:id_"

# endregion
# 推理视频流保存视频文件路劲
DEDUCTION_PROCESS_LOG_VIDEO_PATH = "/log/videos"
# 推理视频流保存推理贞图片路劲
DEDUCTION_PROCESS_LOG_IMAGE_PATH = "/log/images"

# 训练相关配置
TRAIN_ORGMODEL_PATH = "/data/train/model/actionFirst"  # YOLOV8初始训练模型文件存放路径，其余算法是使用网络下载的，所以不用配置
TRAIN_MODEL_SAVE_PATH = "/data/train/model/actionList"  # 训练模型文件存放路径（每次训练记录产生的文件存放路径）
TRAIN_IMAGES_SAVE_PATH = "/data/train/data/images"  # 训练所用图片文件存放路径
TRAIN_VIDEOS_SAVE_PATH = "/data/train/data/videos"  # 训练所用视频文件存放路径
TRAIN_DATASETS_SAVE_PATH = "/data/train/data"  # 训练需要的cocojson、txt、csv等文件存放路径（Java端[也部署在推理服务器上]完成动作和工序编排后，或动作指定了新的启用模型，将信息汇总分别生成cocojson、txt、csv格式，并将这些文件推送到训练服务器）
TRAIN_YOLO_TEMPDATA_PATH = "/data/train/tempData/yoloData"  # YOLOV8训练数据集存放目录

# 服务器参数相关
# sudo -S sshpass -p '123456' rsync -rz -e 'ssh -p 22'  /data/json/1_frames_00000.jpg.json lenovo@192.168.1.188:/data/json/1.json
DEDUCTION_SERVER_IP = "192.168.198.200"  # 推理服务器IP地址
DEDUCTION_SERVER_SSH_PORT: int = 22  # 推理服务器IP地址端口
DEDUCTION_SERVER_SSH_USER = "lenovo"  # 推理服务器ssh账号密码
DEDUCTION_SERVER_SSH_PASSWORD = "123456"  # 推理服务器ssh账号密码
TRAIN_SERVER_IP = "192.168.198.201"  # 训练服务器IP地址
TRAIN_SERVER_SSH_PORT = 22  # 训练服务器ssh端口
TRAIN_SERVER_SSH_USER = "lenovo"  # 训练服务器ssh账号
TRAIN_SERVER_SSH_PASSWORD = "123456"  # 训练服务器ssh密码
PYTHON_STATIC_PATH = "/data"  # PYTHON平台接口地址(内网地址即可，开放权限)
# MASKRCNN_DEFALT_DEVICE = "cuda"  # 默认设备
# TIMESFORMER_DEFALT_DEVICE = "cuda"  # TimeSformer默认设备
# YOLOV8_DEVICE = "cuda"  # 默认设备
MASKRCNN_DEFALT_DEVICE = "cpu"  # 默认设备
TIMESFORMER_DEFALT_DEVICE = "cpu"  # TimeSformer默认设备
YOLOV8_DEVICE = "cpu"  # 默认设备
# REDIS_HOST: str = "192.168.198.200"
# REDIS_PORT: int = 6379
# REDIS_DB_INDEX: int = 1
# REDIS_PASSWORD:str = "redis"

# REDIS_HOST: str = "192.168.1.177"
# REDIS_PORT: int = 6379
# REDIS_DB_INDEX: int = 2
# REDIS_PASSWORD:str = "redis"

REDIS_HOST: str = "127.0.0.1"
# REDIS_HOST: str = "localhost"
REDIS_PORT: int = 6379
REDIS_DB_INDEX: int = 3
# REDIS_PASSWORD:str = "RedIs#749573_"
REDIS_PASSWORD:str = ""
REDIS_MAX_CONN: int = 20
MASKRCNN_DEFALT_THRESH = 0.8  # MASKRCNN默认检测阈值
NN_DEFALT_THRESH = 0.8  # TimeSformer默认检测阈值
YOLOV8_DEFALT_THRESH = 0.8  # YOLOV8默认检测阈值
PROCESS_ONE_TIME = 60  # 推理服务检测时间,正常设为60秒即可，测试机制可以设短一点
PROCESS_TH_PX = "process_th_"  # 推理服务线程ID前缀

# 相关Java接口地址
# JAVA_API_PATH = "http://192.168.198.200:8080"  # JAVA平台接口地址(内网地址即可，开放权限)
JAVA_API_PATH = "http://localhost:8081"  # JAVA平台接口地址(内网地址即可，开放权限)
SEND_LABEL_TO_JAVA = "/tag/labelList/fromPython"  # 将预标记接口返回给Java端
SEND_TRAIN_LOG_TO_JAVA = "/train/trainingLogList/addCirclePassTrainLog"  # 将训练日志返回给Java端
SEND_TRAIN_RESULT_TO_JAVA = "/train/trainingList/circlePassTrainResults"  # 将训练结果返回给Java端
SEND_PROCESS_RESULT_TO_JAVA = "/detect/monitorLogs/getPythonProcess"  # 将推理结果返回给Java端
GET_CAMERA_RETRIEVAL_VO = "/base/cameras/getCameraRetrievalListFromPython"  # 将预标记接口返回给Java端
POST_DEDUCTION_SAVE_VIDEO_JAVA = "/tag/videoList/savePython"  # 将预标的视频地址返回给java端
