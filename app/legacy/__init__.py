"""旧版代码子系统。

包含从 python/ 和 app/services/video_stream_legacy/ 整合的所有旧版功能：
- common/      旧版工具类（config、ajaxResult、redisManager、threadTaskManager 等）
- model/       旧版数据模型（videoInput.py）
- process_videoStream.py  核心视频流处理（增强版）
- extra_service.py        Java 轮询服务
- java_config_client.py   Java API 封装
- rcn/          Mask R-CNN / Faster R-CNN（待补全）
- nn/           TimeSformer 动作识别（待补全）
- yolo_task/    YOLO 独立脚本（待补全）
- frames/       视频帧处理（待补全）

与新版代码完全隔离。通过 PUBLIC_TYPE 开关控制是否启用。
"""

import sys
from pathlib import Path

_LEGACY = Path(__file__).resolve().parent
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))
