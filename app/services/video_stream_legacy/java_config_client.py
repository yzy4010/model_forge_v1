from typing import Optional

from app.services.video_stream_legacy.common.ajaxResult import getJsonWithOutJwt
from app.services.video_stream_legacy.common.config import JAVA_API_PATH, GET_CAMERA_RETRIEVAL_VO
from app.services.video_stream_legacy.model import JavaProcessorConfigResult, ProcessorConfig


def fetch_all_processor_configs_from_java() -> Optional[list[ProcessorConfig]]:
    """调用 Java `getCameraRetrievalListFromPython`，返回全部摄像头推理配置列表。"""
    url = JAVA_API_PATH + GET_CAMERA_RETRIEVAL_VO
    java_data = getJsonWithOutJwt(url, None)
    data = JavaProcessorConfigResult(**java_data)
    if data.code != 200 or data.data is None:
        return None
    return list(data.data)


def fetch_processor_config_from_java(cameras_id: str) -> Optional[ProcessorConfig]:
    url = JAVA_API_PATH + GET_CAMERA_RETRIEVAL_VO
    java_data = getJsonWithOutJwt(url, None)
    data = JavaProcessorConfigResult(**java_data)

    if data.code != 200 or not data.data:
        return None

    for cfg in data.data:
        # 用 camerasId 做匹配
        if getattr(cfg, "camerasId", None) == cameras_id:
            # 这里在 Python 侧赋一个 id，后续 Redis/statusKey 都用它
            # 比如直接用 camerasId 转 int，或用枚举索引
            if not hasattr(cfg, "id") or cfg.id is None:
                try:
                    cfg.id = int(cameras_id)
                except ValueError:
                    cfg.id = 0  # 或别的默认
            return cfg

    return None