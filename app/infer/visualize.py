from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import cv2

def draw_detections(
        frame: Any,
        detections: Iterable[Mapping[str, Any]],
        alias: str = "unknown",  # 显式传入 alias 作为类别名
        title: str | None = None,
) -> Any:
    """
    优化后的目标检测绘制函数。
    统一文本格式为 '类别: 置信度' (例如 vest: 0.91)，并保持自适应清晰度优化。
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # 1. 动态计算基础参数，以适应不同分辨率 (以 1000px 为基准)
    base_scale = min(w, h) / 1000.0
    thickness = max(1, int(2 * base_scale))
    font_scale = max(0.5, 0.6 * base_scale)

    # 2. 绘制标题 (左上角)
    if title:
        title_thickness = max(2, int(3 * base_scale))
        title_scale = max(0.8, 1.0 * base_scale)
        # 绘制带阴影的标题以增强对比度
        cv2.putText(
            overlay, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
            title_scale, (0, 0, 0), title_thickness + 1, cv2.LINE_AA
        )
        cv2.putText(
            overlay, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            title_scale, (0, 255, 0), title_thickness, cv2.LINE_AA
        )

    for det in detections:
        xyxy = det.get("xyxy", [])
        if not isinstance(xyxy, Sequence) or len(xyxy) != 4:
            continue

        x1, y1, x2, y2 = (int(value) for value in xyxy)

        # 3. 统一文本格式：使用传入的 alias
        conf = det.get("conf")
        if isinstance(conf, (int, float)):
            text = f"{alias}: {conf:.2f}"
        else:
            text = f"{alias}"

        # 4. 绘制检测框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # 5. 绘制标签背景和文字
        if text:
            # 计算文字大小
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # 确保标签背景不超出边界 (如果框太靠顶，则向下移动背景)
            padding = int(5 * base_scale)
            bg_y1 = y1 - text_h - baseline - padding
            if bg_y1 < 0:
                bg_y1 = y1  # 如果上方没空间，就贴着框内顶部绘制

            bg_y2 = bg_y1 + text_h + baseline + padding
            bg_x2 = x1 + text_w + padding

            # 绘制实心背景矩形，使文字更清晰
            cv2.rectangle(overlay, (x1, bg_y1), (bg_x2, bg_y2), color, -1)

            # 绘制文字 (黑色文字在绿色背景上最清晰)
            cv2.putText(
                overlay,
                text,
                (x1 + int(padding / 2), bg_y2 - baseline - int(padding / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # 黑色文字
                thickness,
                cv2.LINE_AA,
            )

    return overlay

def _alias_color(alias: str) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    return palette[hash(alias) % len(palette)]


def _parse_bbox(det: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    xyxy = det.get("xyxy")
    if isinstance(xyxy, Sequence) and len(xyxy) == 4:
        x1, y1, x2, y2 = (int(value) for value in xyxy)
        return x1, y1, x2, y2

    bbox = det.get("bbox")
    if isinstance(bbox, Mapping):
        if all(key in bbox for key in ("x1", "y1", "x2", "y2")):
            return (
                int(bbox["x1"]),
                int(bbox["y1"]),
                int(bbox["x2"]),
                int(bbox["y2"]),
            )
        if all(key in bbox for key in ("x", "y", "w", "h")):
            x1 = int(bbox["x"])
            y1 = int(bbox["y"])
            x2 = x1 + int(bbox["w"])
            y2 = y1 + int(bbox["h"])
            return x1, y1, x2, y2

    if all(key in det for key in ("x", "y", "w", "h")):
        x1 = int(det["x"])
        y1 = int(det["y"])
        x2 = x1 + int(det["w"])
        y2 = y1 + int(det["h"])
        return x1, y1, x2, y2

    return None


def draw_alias_detections(
        frame: Any,
        results: Mapping[str, Mapping[str, Any]],
) -> Any:
    """
    优化后的 draw_alias_detections 函数。
    通过动态计算线宽、字体缩放，并为文字添加背景色块，确保在不同分辨率下都清晰可见。
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # 1. 动态计算基础参数 (以 1000px 为基准缩放)
    base_scale = min(w, h) / 1000.0
    thickness = max(1, int(2 * base_scale))
    font_scale = max(0.5, 0.6 * base_scale)

    for alias, result in results.items():
        detections = result.get("detections", []) if isinstance(result, Mapping) else []
        # 假设 _alias_color 是外部定义的函数
        color = _alias_color(alias)

        for det in detections:
            if not isinstance(det, Mapping):
                continue

            # 假设 _parse_bbox 是外部定义的函数
            coords = _parse_bbox(det)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            conf = det.get("conf")

            # 构造显示文本
            text = f"{alias}: {conf:.2f}" if isinstance(conf, (int, float)) else f"{alias}"

            # 2. 绘制检测框
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            # 3. 绘制带背景的标签文字
            if text:
                # 计算文字占据的尺寸
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                # 确定背景框的位置（尽量放在框上方，如果太靠顶则放在框内）
                padding = int(5 * base_scale)
                bg_y1 = y1 - text_h - baseline - padding
                if bg_y1 < 0:
                    bg_y1 = y1

                bg_y2 = bg_y1 + text_h + baseline + padding
                bg_x2 = x1 + text_w + padding

                # 绘制实心背景矩形
                cv2.rectangle(overlay, (x1, bg_y1), (bg_x2, bg_y2), color, -1)

                # 绘制文字
                # 为了清晰，根据背景颜色亮度决定文字用黑色还是白色
                # 简单处理：如果背景是绿色/浅色，用黑色；如果是深色，用白色
                # 这里默认使用对比色或固定白色/黑色
                text_color = (255, 255, 255)  # 默认白色
                # 如果颜色亮度较高，则使用黑色文字
                if isinstance(color, (list, tuple)) and len(color) >= 3:
                    brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                    if brightness > 128:
                        text_color = (0, 0, 0)

                cv2.putText(
                    overlay,
                    text,
                    (x1 + int(padding / 2), bg_y2 - baseline - int(padding / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    thickness,
                    cv2.LINE_AA,
                )

    return overlay
