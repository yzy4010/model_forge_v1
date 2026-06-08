import os, json, time, torch, platform
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from rcn.train_utils import coco_remove_images_without_annotations, convert_coco_poly_mask
from common.runScript import *
from common.config import *

osName = platform.system().lower()


class UTF8COCO(COCO):
    """支持 UTF-8 编码的自定义 COCO 类"""
    
    def __init__(self, annotation_file=None):
        """
        重写初始化方法，支持 UTF-8 编码
        
        参数:
            annotation_file (str, optional): COCO JSON 文件路径
        """
        if annotation_file is not None:
            # 安全加载 JSON 文件
            with open(annotation_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            # 初始化 COCO 数据集
            self.dataset = dataset
            self.createIndex()

class CocoDetection(data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        dataset (string): train or val.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, anno_name, dataset="train", transforms=None):
        super(CocoDetection, self).__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = root
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = anno_name + ".json"
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        # self.mode = dataset
        self.transforms = transforms
        self.coco = UTF8COCO(self.anno_path)

        # 获取coco数据索引与类别名称的关系
        # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        min_index = min(data_classes.keys())  # 90
        max_index = max(data_classes.keys())  # 90
        coco_classes = {}
        cut:int=1 if min_index == 0 else 0     #避免category_id从0开始
        for k in range(1, max_index + 1+cut):
            if k-cut in data_classes:
                coco_classes[k] = data_classes[k-cut]
            else:
                coco_classes[k] = "N/A"

        if dataset == "train":
            json_str = json.dumps(coco_classes, ensure_ascii=False)
            with open(anno_name + "_classes.json", "w",encoding="utf-8") as f:
                f.write(json_str)
                # or osName=="darwin"  苹果   windows
            if (osName == "linux"):
                # 将此文件推送到推理服务器
                time.sleep(2)
                # DEDUCTION_SERVER_SSH.split("@")[0]+":"+TRAIN_DATASETS_SAVE_PATH+"/"+anno_name+"_classes.json",local_password=DEDUCTION_SERVER_SSH.split("@")[1])
                rsync_push(local_path=anno_name + "_classes.json",
                            remote_path=TRAIN_DATASETS_SAVE_PATH,
                            remoteServer=linuxServer(
                                isNeedSudo=True,
                                ip=DEDUCTION_SERVER_IP,
                                port=DEDUCTION_SERVER_SSH_PORT,
                                user=DEDUCTION_SERVER_SSH_USER,
                                password=DEDUCTION_SERVER_SSH_PASSWORD),
                            local_password=TRAIN_SERVER_SSH_PASSWORD,
                            shell=True)

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*[item for item in batch if item is not None]))

# if __name__ == '__main__':
#     train = CocoDetection("/data/coco2017", dataset="train")
#     print(len(train))
#     t = train[0]
