# import datetime
from typing import Optional
from pydantic import BaseModel


class actionInfo(BaseModel):
    # 动作模型id   
    id: str
    # 动作模型算法类型  1：Mask R-CNN  2：yolov8  3：YoLact  4：Torch.nn
    modelType: str


class videoActionInfo(BaseModel):
    '''
    @staticmethod
    def getAllBlanks(book:xls.Workbook):
        refs=[]
        sheet=book["填空题"]
        point:int=int(str(sheet["F1"].value))
        row_num=sheet.max_row
        rows=sheet["A3":"D%d"%row_num]
        for row in rows:
            refs.append(BoolInfo(
                int(str(row[0].value)),
                str(row[1].value),
                point,
                str(row[2].value),
                str(row[3].value)
            ))
        return refs

    def setStudValue(self, value:str,time:datetime):
        self.StudValue = value
        self.StudTime = time
    '''
    # 线程的Id
    guId: str
    # 动作集合
    actions: list[actionInfo]
    # 视频流或者视频文件地址
    videoFile: str
    # 视频文件的类型 1 视频流 2 视频文件地址
    videoType: str


class labelConfig():
    def __init__(self, id: str, modelType: str, videoFile: str, videoType: str, guid: str, isGetNoLabel: bool):  # 构造函数，类实例化是自动执行
        # 初始化实例属性
        self.id = id
        self.modelType = modelType
        self.videoFile = videoFile
        self.videoType = videoType
        self.guid = guid
        self.isGetNoLabel = isGetNoLabel  

    # 动作模型id
    id: str
    # 动作模型算法类型  1：Mask R-CNN  2：yolov8  3：YoLact  4：Torch.nn
    modelType: str
    # 视频流或者视频文件地址
    videoFile: str
    # 视频文件的类型 1 视频流 2 视频文件地址
    videoType: str
    # 线程ID
    guid: str
    # 是否回传五标记图片
    isGetNoLabel: bool = False


class maskRcnnTrainConfig():
    def __init__(self, actionId: str, trainLogId: str, modelType: str = "1", epochs: int = 20, batch: int = 4,
                 lr: float = 0.004, amp: bool = False, momentum: float = 0.9, weightDecay: float = 0.0004,
                 resume: bool = False, startEpoch: int = 0,
                 ):  # 构造函数，类实例化是自动执行
        # 初始化实例属性
        self.actionId = actionId
        self.trainLogId = trainLogId
        self.modelType = modelType
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.amp = amp
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.resume = resume
        self.startEpoch = startEpoch

    # 动作模型id
    actionId: str
    # 动作模型算法类型  1：Mask R-CNN  2：yolov8  3：YoLact  4：Torch.nn
    modelType: str
    # 训练轮数
    epochs: int
    # 批次大小
    batch: int
    # 学习率
    lr: float
    # 是否使用GPU混合精度训练
    amp: bool
    # SGD的momentum参数
    momentum: float
    # SGD的weight_decay参数
    weightDecay: float
    # 恢复轮数
    startEpoch: int
    # 是否恢复训练
    resume: bool


# region yolo训练
# modelConfig  模型训练参数
class ModelConfig(BaseModel):
    # # 训练数据集
    # data: str
    # 训练轮数
    epochs: int=20
    # 输入图像尺寸 默认640
    imgSiz: int = 640
    # 批次大小
    batch: int=4

    # 学习率-maskcnn-0.004
    lr: float = 0.004
    # SGD的momentum参数-maskcnn-0.9
    momentum: float = 0.9
    # SGD的weight_decay参数-maskcnn-0.0004
    weightDecay: float = 0.0004


class TrainRecordInfo(BaseModel):
    # 训练记录 id
    trainingRecordListId: str
    # 训练 id
    trainingListId: str = ""
    # 训练名称
    tranName: str = ""
    # 动作 ID
    actionId: str
    # 动作名称
    actionName: str = ""


class StopTrainInput(BaseModel):
    # 训练记录 id
    trainingRecordListId: str


class TrainParameter(BaseModel):
    # 动作模型算法类型  1：Mask R-CNN  2：yolov8  3：YoLact  4：Torch.nn
    modelType: Optional[str] = None
    modelConfig: ModelConfig
    trainRecords: list[TrainRecordInfo]


# endregion


class linuxServer():
    def __init__(self, isNeedSudo: bool = False, ip: str = "localhost", port: int = 22, user: str = "root",
                 password: str = None, keyPath: str = None):  # 构造函数，类实例化是自动执行
        # 初始化实例属性
        self.isNeedSudo = isNeedSudo
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.keyPath = keyPath

    # 是否需要sudo前缀
    isNeedSudo: bool = False
    # 服务IP
    ip: str = "localhost"
    # SSH端口
    port: int = 22
    # 用户名
    user: str = "root"
    # 密码
    password: str = None
    # 私钥文件路径
    keyPath: str = None


class LabelInfo(BaseModel):
    id: Optional[str] = None
    labelName: Optional[str] = None
    isMain: Optional[str] = None


class BeatInfo(BaseModel):
    id: Optional[str] = None
    beatName: Optional[str] = None
    # 节拍时间上限
    beatUpperTime: Optional[int] = None
    # 节拍时间下限
    beatLowerTime: Optional[int] = None
    labelList: Optional[list[LabelInfo]] = None


class ProcessorConfig(BaseModel):
    # 推理配置
    id: int = -1
    actionId: Optional[str] = None
    actionName: Optional[str] = None
    actionType: Optional[str] = None
    camerasId: Optional[str] = None
    camerasName: Optional[str] = None
    camerasStatus: Optional[str] = None
    monitorConfigId: Optional[str] = None
    minThresh: Optional[float] = None
    rtspurl: Optional[str] = None
    ffurl: Optional[str] = None
    httpurl: Optional[str] = None
    workstationId: Optional[str] = None
    workstationName: Optional[str] = None
    beatList: Optional[list[BeatInfo]] = None


class JavaProcessorConfigResult(BaseModel):
    code: int = 200,
    msg: str = "success",
    data: list[ProcessorConfig] = None
