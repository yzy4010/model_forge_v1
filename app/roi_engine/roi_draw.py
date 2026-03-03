import cv2
import numpy as np
import os
from PIL import ImageFont
from PIL import Image, ImageDraw

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