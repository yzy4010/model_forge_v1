import os
import time
import json

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # polygons = []
    #选择面积最大的轮廓
    main_contour = max(contours, key=cv2.contourArea)
    # tolerance: 简化程度(值越大点越少)
    # tolerance = 0.5
    # epsilon = tolerance * cv2.arcLength(main_contour, True)
    # main_contour = cv2.approxPolyDP(main_contour, epsilon, True)

    # for contour in contours:
    #     if len(contour) >= 3:  # 至少三个点才能形成多边形
    #         polygons.append(contour.flatten().tolist())
    polygon = main_contour.flatten().tolist()
    # print(f"polygon-----{polygon}")
    return polygon


def masks_to_coco_polygon(masks):
    """
    将多个二值掩码转换为 COCO 的 Polygon 格式 segmentation 数据
    :param masks: shape [N, H, W], bool 类型的掩码数组
    :return: list of list, 每个元素是坐标列表组成的多边形
    """
    segmentations = []
    for mask in masks:
        polygon = mask_to_polygon(mask)
        segmentations.append(polygon)
    # print(f"segmentations-----{segmentations}")
    return segmentations


def main():
    num_classes = 8  # 不包含背景
    box_thresh = 0.3 # 检测阈值
    weights_path = "/data/train/model/actionTrue/visionai-newest_1.pth"
    img_path = "/data/train/data/images/frame_0000.jpg"
    label_json_path = "/data/train/model/actionTrue/visionai-newest_1_classes.json"

    # get devices
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r',encoding="utf-8") as json_file:
        category_index:dict = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        prediction = model(img.to(device))[0]
        # predictions = model(img.to(device))
        # outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in predictions]

        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))
        # prediction=predictions[0]
        predict_boxes = prediction["boxes"].to("cpu").numpy()
        predict_classes = prediction["labels"].to("cpu").numpy()
        predict_scores = prediction["scores"].to("cpu").numpy()
        predict_mask_org = prediction["masks"]
        predict_mask = np.squeeze(predict_mask_org.cpu().numpy(), axis=1)  # [batch, 1, h, w] -> [batch, h, w]
        # predict_mask = np.where(predict_mask > 0.7, True, False)
        # print("predict_mask:",predict_mask)
        maskss = np.where(predict_mask > box_thresh, True, False)
        segmentations=masks_to_coco_polygon(maskss)

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return
        results = []
        for i in range(len(predict_scores)):
            # 解析检测信息
            label_id = str(predict_classes[i])
            box = predict_boxes[i].astype(int)


            score = predict_scores[i]
            label = category_index.get(label_id, 'N/A')

            # 记录结果数据
            results.append({
                "label": label,
                "labelid": label_id,
                "score": float(score),
                "bbox": box.tolist(),
                "segmentation": [segmentations[i]]
            })
        print(f"检测结果：{results}")  
        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             box_thresh=box_thresh,
                             mask_thresh=box_thresh,
                             category_index=category_index,
                             line_thickness=3,
                             font='/data/deduction/data/simhei.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()

