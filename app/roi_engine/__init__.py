#!/usr/bin/env python3
"""
ROI 模块初始化文件

导入必要的模块和类
"""

from app.roi_engine.roi_protocol import ROI, Detection
from app.roi_engine.geometry_utils import point_in_polygon
from app.roi_engine.roi_engine import ROIEngine

__all__ = ["ROI", "Detection", "point_in_polygon", "ROIEngine"]
