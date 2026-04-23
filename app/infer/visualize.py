from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fonts", "simsunb.ttf")),
    "/usr/share/fonts/myfonts/truetype/STSONG.TTF",
    "/usr/share/fonts/myfonts/truetype/SIMYOU.TTF",
    "/usr/share/fonts/myfonts/truetype/msyh.ttc",
    "/usr/share/fonts/myfonts/truetype/ARIALUNI.TTF",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Light.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Black.ttc",
    "/usr/share/fonts/google-noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
]
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}
_FONT_SOURCE_BY_SIZE: dict[int, str] = {}
_FONT_LOGGED = False


def _get_font(size_px: int) -> ImageFont.FreeTypeFont:
    global _FONT_LOGGED
    size_px = max(12, int(size_px))
    cached = _FONT_CACHE.get(size_px)
    if cached is not None:
        return cached
    for p in _FONT_PATHS:
        if os.path.exists(p):
            font = ImageFont.truetype(p, size_px)
            _FONT_CACHE[size_px] = font
            _FONT_SOURCE_BY_SIZE[size_px] = p
            if not _FONT_LOGGED:
                print(f"[visualize] PIL font selected: {p}")
                _FONT_LOGGED = True
            return font
    # 兜底，避免字体缺失时抛异常
    font = ImageFont.load_default()
    _FONT_CACHE[size_px] = font
    _FONT_SOURCE_BY_SIZE[size_px] = "PIL_DEFAULT"
    if not _FONT_LOGGED:
        print("[visualize] PIL font fallback: PIL_DEFAULT")
        _FONT_LOGGED = True
    return font


def _draw_text_with_bg(
        frame: Any,
        text: str,
        left: int,
        top: int,
        bg_color_bgr: tuple[int, int, int],
        text_color_bgr: tuple[int, int, int],
        font_size_px: int,
) -> Any:
    """用 PIL 绘制中文文本，并保持 OpenCV 的 BGR 图像流。"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size_px)

    # 使用 textbbox 获取真实文本尺寸，兼容中文。
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, r - l)
    text_h = max(1, b - t)
    pad = max(2, int(font_size_px * 0.22))

    x1 = max(0, int(left))
    y1 = max(0, int(top))
    x2 = x1 + text_w + pad * 2
    y2 = y1 + text_h + pad * 2

    # 先画背景框，再画文字（颜色需从 BGR 转 RGB）。
    bg_rgb = (bg_color_bgr[2], bg_color_bgr[1], bg_color_bgr[0])
    text_rgb = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0])
    draw.rectangle((x1, y1, x2, y2), fill=bg_rgb)
    draw.text((x1 + pad, y1 + pad), text, font=font, fill=text_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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
        title_font_px = max(16, int(26 * base_scale))
        overlay = _draw_text_with_bg(
            overlay,
            title,
            left=10,
            top=8,
            bg_color_bgr=(0, 80, 0),
            text_color_bgr=(255, 255, 255),
            font_size_px=title_font_px,
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
            font_px = max(14, int(22 * base_scale))
            label_top = y1 - (font_px + 10)
            if label_top < 0:
                label_top = y1
            overlay = _draw_text_with_bg(
                overlay,
                text,
                left=x1,
                top=label_top,
                bg_color_bgr=color,
                text_color_bgr=(0, 0, 0),
                font_size_px=font_px,
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
            # 优先使用模型输出 label（来自权重 names），避免 alias 链路编码异常导致方框。
            label = det.get("label")
            if not isinstance(label, str) or not label.strip():
                label = alias
            # 过滤不可见控制字符，避免渲染异常
            safe_label = "".join(ch for ch in label if ch.isprintable()).strip() or "unknown"
            text = f"{safe_label}: {conf:.2f}" if isinstance(conf, (int, float)) else safe_label

            # 2. 绘制检测框
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            # 3. 绘制带背景的标签文字
            if text:
                text_color = (255, 255, 255)  # 默认白色
                # 如果颜色亮度较高，则使用黑色文字
                if isinstance(color, (list, tuple)) and len(color) >= 3:
                    brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                    if brightness > 128:
                        text_color = (0, 0, 0)
                font_px = max(14, int(22 * base_scale))
                label_top = y1 - (font_px + 10)
                if label_top < 0:
                    label_top = y1
                overlay = _draw_text_with_bg(
                    overlay,
                    text,
                    left=x1,
                    top=label_top,
                    bg_color_bgr=color,
                    text_color_bgr=text_color,
                    font_size_px=font_px,
                )

    return overlay
