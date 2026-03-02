#!/usr/bin/env python3
"""
ROI 引擎

实现 ROIEngine 类，用于将检测结果映射到 ROI 标签
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional
from app.roi_engine.roi_protocol import ROI, Detection
from app.roi_engine.geometry_utils import point_in_polygon, compute_bbox_center
import threading


class ROIEngine:
    """
    ROI 引擎类
    
    该引擎处理检测结果，并根据检测结果的中心点是否位于任何预定义的多边形内
    来添加 ROI 标签。
    
    属性:
        camera_id (str): 相机的唯一标识符
        config_version (str): 配置版本
        resolution (Dict[str, int]): 相机分辨率 (width, height)
        rois (List[ROI]): ROI 配置列表
        _enabled_rois (List[Dict]): 启用的 ROI 列表，包含预计算的多边形
        _lock (threading.RLock): 线程锁，确保线程安全
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        使用配置初始化 ROI 引擎
        
        参数:
            config (Dict[str, Any]): 运行时配置 JSON
                {
                    "camera_id": str,
                    "config_version": str,
                    "resolution": {"width": int, "height": int},
                    "rois": [
                        {
                            "roi_id": str,
                            "name": str,
                            "semantic_tag": str,
                            "enabled": bool,
                            "geometry": {
                                "points": [[x1, y1], [x2, y2], ...]
                            }
                        }
                    ]
                }
        """
        self._lock = threading.RLock()
        self.camera_id = config.get("camera_id")
        self.config_version = config.get("config_version")
        self.resolution = config.get("resolution", {"width": 1920, "height": 1080})
        
        # 解析 ROI 配置并预计算多边形
        roi_list = config.get("rois", [])
        self.rois = []
        self._enabled_rois = []
        
        with self._lock:
            for roi_dict in roi_list:
                roi = ROI(
                    roi_id=roi_dict.get("roi_id"),
                    name=roi_dict.get("name"),
                    semantic_tag=roi_dict.get("semantic_tag"),
                    enabled=roi_dict.get("enabled", False),
                    geometry=roi_dict.get("geometry", {})
                )
                self.rois.append(roi)
            
            # 过滤出启用的 ROI 并预计算多边形
            self._process_rois()
    
    def _process_rois(self):
        """
        处理 ROI 配置并预计算多边形以提高性能
        """
        self._enabled_rois = []
        for roi in self.rois:
            if roi.enabled and roi.geometry.get("points"):
                points = roi.geometry.get("points", [])
                if len(points) >= 3:  # 多边形至少需要 3 个点
                    # 预计算多边形 numpy 数组
                    polygon_array = np.array(points, dtype=np.int32)
                    self._enabled_rois.append({
                        "roi_id": roi.roi_id,
                        "semantic_tag": roi.semantic_tag,
                        "polygon": polygon_array
                    })
    
    def update(self, config: Dict[str, Any]):
        """
        更新引擎配置（热重载）
        
        参数:
            config (Dict[str, Any]): 新的运行时配置 JSON
        """
        with self._lock:
            self.camera_id = config.get("camera_id", self.camera_id)
            self.config_version = config.get("config_version", self.config_version)
            self.resolution = config.get("resolution", self.resolution)
            
            # 解析 ROI 配置
            roi_list = config.get("rois", [])
            self.rois = []
            for roi_dict in roi_list:
                roi = ROI(
                    roi_id=roi_dict.get("roi_id"),
                    name=roi_dict.get("name"),
                    semantic_tag=roi_dict.get("semantic_tag"),
                    enabled=roi_dict.get("enabled", False),
                    geometry=roi_dict.get("geometry", {})
                )
                self.rois.append(roi)
            
            # 过滤出启用的 ROI 并预计算多边形
            self._process_rois()

    def apply(self, detections):
        """
        支持 dict 结构的 detections
        """

        with self._lock:
            enabled_rois = self._enabled_rois.copy()

        processed = []

        for detection in detections:

            # ================= 兼容 dict =================
            if isinstance(detection, dict):
                xyxy = detection.get("xyxy")
                if not xyxy:
                    processed.append(detection)
                    continue

                # 计算中心点
                x1, y1, x2, y2 = xyxy
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # 初始化 roi_tags
                detection.setdefault("roi_tags", [])

                for roi in enabled_rois:
                    result = cv2.pointPolygonTest(roi["polygon"], center, False)
                    if result >= 0:
                        tag = roi["semantic_tag"]
                        if tag not in detection["roi_tags"]:
                            detection["roi_tags"].append(tag)

                processed.append(detection)

            # ================= 如果还是老 Detection 类 =================
            else:
                center = compute_bbox_center(detection.xxyy)

                for roi in enabled_rois:
                    result = cv2.pointPolygonTest(roi["polygon"], center, False)
                    if result >= 0:
                        semantic_tag = roi["semantic_tag"]
                        if semantic_tag not in detection.roi_tags:
                            detection.roi_tags.append(semantic_tag)

                processed.append(detection)

        return processed


# 示例用法
if __name__ == "__main__":
    # 示例配置
    sample_config = {
        "camera_id": "cam_001",
        "config_version": "1.0",
        "resolution": {
            "width": 1920,
            "height": 1080
        },
        "rois": [
            {
                "roi_id": "roi_001",
                "name": "危险区域",
                "semantic_tag": "danger_zone",
                "enabled": True,
                "geometry": {
                    "points": [[100, 100], [300, 100], [300, 300], [100, 300]]
                }
            },
            {
                "roi_id": "roi_002",
                "name": "区域 A",
                "semantic_tag": "area_A",
                "enabled": True,
                "geometry": {
                    "points": [[200, 200], [400, 200], [400, 400], [200, 400]]
                }
            }
        ]
    }
    
    # 初始化 ROI 引擎
    engine = ROIEngine(sample_config)
    
    # 示例检测结果
    sample_detections = [
        Detection(
            cls="person",
            conf=0.89,
            xxyy=[150, 150, 250, 250]  # 中心点 (200, 200) - 在两个 ROI 内
        ),
        Detection(
            cls="car",
            conf=0.92,
            xxyy=[350, 350, 450, 450]  # 中心点 (400, 400) - 仅在区域 A 内
        ),
        Detection(
            cls="dog",
            conf=0.75,
            xxyy=[600, 600, 700, 700]  # 中心点 (650, 650) - 在所有 ROI 外
        )
    ]
    
    # 应用 ROI 标签
    processed = engine.apply(sample_detections)
    
    # 打印结果
    print("处理后的检测结果:")
    for i, det in enumerate(processed):
        print(f"检测结果 {i+1}:")
        print(f"  类别: {det.cls}")
        print(f"  置信度: {det.conf}")
        print(f"  边界框: {det.xxyy}")
        print(f"  ROI 标签: {det.roi_tags}")
        print()
    
    # 热重载示例
    print("=== 热重载示例 ===")
    new_config = sample_config.copy()
    new_config["config_version"] = "1.1"
    new_config["rois"].append({
        "roi_id": "roi_003",
        "name": "新区域",
        "semantic_tag": "new_area",
        "enabled": True,
        "geometry": {
            "points": [[450, 100], [650, 100], [650, 300], [450, 300]]
        }
    })
    
    engine.update(new_config)
    processed_after_update = engine.apply(sample_detections)
    
    print("热重载后的检测结果:")
    for i, det in enumerate(processed_after_update):
        print(f"检测结果 {i+1}:")
        print(f"  类别: {det.cls}")
        print(f"  置信度: {det.conf}")
        print(f"  边界框: {det.xxyy}")
        print(f"  ROI 标签: {det.roi_tags}")
        print()
