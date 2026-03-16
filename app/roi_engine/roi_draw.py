import cv2
import numpy as np
import os
from PIL import ImageFont
from PIL import Image, ImageDraw
import logging
logger = logging.getLogger("model_forge.roi_engine_roi_draw")

FONT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "fonts",
        "simhei.ttf"
    )
)

FONT = ImageFont.truetype(FONT_PATH, 24)


def draw_rois(frame, roi_config, event_results=None):
    """
    画 ROI，多边形
    如果某 ROI 被命中，则变为红色
    """
    if not roi_config:
        return frame

    rois = roi_config.get("rois", [])
    hit_tags = set()

    # ================= 判断哪些 ROI 被命中 =================
    if event_results:
        for alias_result in event_results.values():
            detections = alias_result.get("detections", [])
            for det in detections:
                if 'roi_tags' in det:
                    for tag in det['roi_tags']:
                        hit_tags.add(tag)

    # =======================================================

    for roi in rois:
        if not roi.get("enabled", True):
            continue

        points = roi.get("geometry", {}).get("points", [])
        if not points:
            continue

        pts = np.array(points, dtype=np.int32)
        tag = roi.get("semantic_tag")

        # 如果命中 → 红色，否则绿色
        if tag in hit_tags:
            color = (0, 0, 255)   # 红色
            thickness = 2  # 命中加粗
        else:
            color = (0, 255, 0) # 绿色
            thickness = 2  # 不命中线条较细

        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness
        )

        name = roi.get("name", "")
        if name:
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)

            draw.text(
                (int(pts[0][0]), int(pts[0][1]) - 25),
                name,
                font=FONT,
                fill=(color[2], color[1], color[0])  # BGR → RGB
            )

            frame = np.array(img_pil)

    return frame

def draw_rois_by_rule(frame, roi_config, triggered_rois: set):
    """
    根据规则触发状态绘制 ROI
    triggered_rois: 真正导致规则成立的语义标签集合
    """
    if not roi_config:
        return frame

    rois = roi_config.get("rois", [])

    for roi in rois:
        if not roi.get("enabled", True):
            continue

        points = roi.get("geometry", {}).get("points", [])
        if not points:
            continue

        pts = np.array(points, dtype=np.int32)
        tag = roi.get("semantic_tag")

        # 判断变色：只有该区域触发了业务规则，才变红
        if tag in triggered_rois:
            color = (0, 0, 255)   # 红色 (BGR)
            thickness = 3         # 触发时加粗线条
        else:
            color = (0, 255, 0)   # 绿色
            thickness = 2         # 未触发保持细线

        # 绘制多边形
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

        # 绘制名称
        name = roi.get("name", "")
        if name:
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            # 文字颜色跟随线条颜色
            draw.text(
                (int(pts[0][0]), int(pts[0][1]) - 25),
                name,
                font=FONT,
                fill=(color[2], color[1], color[0])
            )
            frame = np.array(img_pil)

    return frame