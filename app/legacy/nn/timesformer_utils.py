import cv2
from PIL import Image
from torchvision.transforms import *
from common.config import *


def keep_aspect_resize(frame, target_size=224):
    """保持宽高比的缩放（短边匹配target_size），截取中间部分"""
    h, w = frame.shape[:2]
    scale = target_size / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

class ToPILWrapper:
    """将numpy数组转为PIL Image"""
    def __call__(self, img):
        return Image.fromarray(img)

class SmartResizePad:
    """保持宽高比缩放+边缘复制填充"""
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # 计算填充量
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        return cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_REPLICATE  # 关键：使用边缘像素填充而非黑边
        )