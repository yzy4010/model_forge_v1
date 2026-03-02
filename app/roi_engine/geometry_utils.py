#!/usr/bin/env python3
"""
几何工具

实现点-in-多边形功能，确保高效处理
"""

import numpy as np
import cv2
from typing import List, Tuple


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    检查点是否在多边形内
    
    参数:
        point (Tuple[float, float]): 点坐标 (x, y)
        polygon (List[List[float]]): 多边形的点列表 [[x1, y1], [x2, y2], ...]
    
    返回:
        bool: 如果点在多边形内或边缘上，返回 True，否则返回 False
    """
    # 将多边形转换为 numpy 数组
    polygon_array = np.array(polygon, dtype=np.int32)
    
    # 使用 OpenCV 的 pointPolygonTest
    # 返回值:
    #   正数: 点在多边形内
    #   零: 点在多边形边缘上
    #   负数: 点在多边形外
    result = cv2.pointPolygonTest(polygon_array, point, False)
    
    # 如果结果 >= 0，表示点在多边形内或边缘上
    return result >= 0


def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    计算边界框的中心点
    
    参数:
        bbox (List[float]): 边界框坐标 [x1, y1, x2, y2]
    
    返回:
        Tuple[float, float]: 中心点坐标 (x, y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)
