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


def draw_rois(frame, roi_config, roi_status=None):
    """
    画 ROI，多边形
    如果某 ROI 被命中（根据 roi_status），则变为红色，否则绿色。
    """
    if not roi_config:
        return frame  # 如果没有 ROI 配置，直接返回原始帧

    rois = roi_config.get("rois", [])  # 获取所有的 ROI 配置

    # 如果提供了 roi_status，我们会使用它来判断哪些 ROI 区域需要变色
    for roi in rois:
        if not roi.get("enabled", True):  # 如果 ROI 未启用，跳过
            continue

        points = roi.get("geometry", {}).get("points", [])
        if not points:  # 如果没有 ROI 点，跳过
            continue

        pts = np.array(points, dtype=np.int32)
        tag = roi.get("semantic_tag")

        # 使用 roi_status 来决定是否变色
        if roi_status and tag in roi_status:
            status = roi_status[tag]  # 获取当前 ROI 的状态
            if status == "alert":  # 如果是警报状态
                color = (0, 0, 255)  # 红色
                thickness = 2  # 加粗线条
            else:
                color = (0, 255, 0)  # 绿色
                thickness = 2  # 普通线条
        else:
            color = (0, 255, 0)  # 默认绿色
            thickness = 2

        # 绘制多边形 ROI 区域
        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness
        )

        # 如果 ROI 配置有名称，显示名称
        name = roi.get("name", "")
        if name:
            img_pil = Image.fromarray(frame)  # 转换为 PIL 图像以便绘制文字
            draw = ImageDraw.Draw(img_pil)

            # 在多边形区域上方绘制 ROI 名称
            draw.text(
                (int(pts[0][0]), int(pts[0][1]) - 25),  # 文字位置在多边形的左上角
                name,
                font=FONT,  # 请确保 FONT 已定义
                fill=(color[2], color[1], color[0])  # BGR → RGB
            )

            frame = np.array(img_pil)  # 再转换回 NumPy 数组

    return frame  # 返回绘制了 ROI 的帧