import cv2
import numpy as np


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
            for det in alias_result.get("detections", []):
                for tag in det.get("roi_tags", []):
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

        # 如果命中 → 红色，否则黄色
        if tag in hit_tags:
            color = (0, 0, 255)   # 红色
            thickness = 3
        else:
            color = (0, 255, 255) # 黄色
            thickness = 2

        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness
        )

        name = roi.get("name", "")
        if name:
            cv2.putText(
                frame,
                name,
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return frame