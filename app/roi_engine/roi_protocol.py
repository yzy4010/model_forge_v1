#!/usr/bin/env python3
"""
ROI 协议定义

定义 ROI 和 Detection 的输入协议结构
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ROI:
    """
    ROI（感兴趣区域）类
    
    属性:
        roi_id (str): ROI 的唯一标识符
        name (str): ROI 的名称
        semantic_tag (str): ROI 的语义标签
        enabled (bool): 是否启用该 ROI
        geometry (Dict[str, Any]): ROI 的几何信息
            {
                "points": [[x1, y1], [x2, y2], ...]  # 多边形的点
            }
    """
    roi_id: str
    name: str
    semantic_tag: str
    enabled: bool
    geometry: Dict[str, Any]


@dataclass
class Detection:
    """
    检测结果类
    
    属性:
        cls (str): 检测类别
        conf (float): 检测置信度
        xxyy (List[float]): 边界框坐标 [x1, y1, x2, y2]
        roi_tags (List[str]): ROI 标签列表
    """
    cls: str
    conf: float
    xxyy: List[float]
    roi_tags: List[str] = None
    
    def __post_init__(self):
        """
        初始化后处理
        """
        if self.roi_tags is None:
            self.roi_tags = []
