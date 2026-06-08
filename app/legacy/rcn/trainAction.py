import os,platform
import datetime
import threading

import torch
from torchvision.ops.misc import FrozenBatchNorm2d

# import torchvision.transforms as transforms
from common.redisManager import RedisManager
import rcn.transforms as transforms
from .network_files import MaskRCNN
from .backbone import resnet50_fpn_backbone
from .my_dataset_coco import CocoDetection
from .my_dataset_voc import VOCInstances
from .train_utils import train_eval_utils as utils
from .train_utils import GroupedBatchSampler, create_aspect_ratio_groups

from common.config import *
from model.videoInput import *
from common.ajaxResult import *
from common.runScript import *
import asyncio


def create_model(coco_classes, actionId,actionWay="1"):
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)
    # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d,
    #                                  trainable_layers=3)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet50_fpn_backbone(pretrain_path=TRAIN_ORGMODEL_PATH+"/resnet50-0676ba61.pth", trainable_layers=1)
    num_classes=len(coco_classes)+1
    print("num_classes-------",num_classes)
    print("coco_classes-------",coco_classes)
    model = MaskRCNN(backbone, num_classes=num_classes)
    actionModelPath = DEDUCTION_MODEL_PATH+"/"+actionId+"_"+actionWay+".pth"

    if os.path.exists(actionModelPath)==False:
        actionModelPath = TRAIN_ORGMODEL_PATH+"/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    else:
        weights_dict_test = torch.load(actionModelPath, map_location="cpu")
        if "coco_classes" in weights_dict_test:
            if weights_dict_test["coco_classes"]!=coco_classes:
                print("actionModelPath的类别与当前类别不一致，请重新训练模型！")
                actionModelPath = TRAIN_ORGMODEL_PATH+"/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        else:
            actionModelPath = TRAIN_ORGMODEL_PATH+"/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    print("actionModelPath预加载模型文件：-------",actionModelPath)
    # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    weights_dict = torch.load(actionModelPath, map_location="cpu")
    for k in list(weights_dict.keys()):
        if ("box_predictor" in k) or ("mask_fcn_logits" in k):
            del weights_dict[k]

    model.load_state_dict(weights_dict, strict=False)
    return model


def startTrain(config:maskRcnnTrainConfig,redis:RedisManager=None):
    device = torch.device(MASKRCNN_DEFALT_DEVICE if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    # now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S"){now}
    det_results_file = f"{TRAIN_MODEL_SAVE_PATH}/{config.trainLogId}_det_results.txt"
    seg_results_file = f"{TRAIN_MODEL_SAVE_PATH}/{config.trainLogId}_seg_results.txt"

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                    # transforms.Resize((450, 800)),
                                    transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = TRAIN_IMAGES_SAVE_PATH

    # load train data set
    # coco2017 -> annotations -> instances_train2017.json
    train_dataset = CocoDetection(data_root, TRAIN_DATASETS_SAVE_PATH+"/"+config.actionId,"train",data_transform["train"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])
    train_sampler = None
    batch_size = config.batch
    train_batch_sampler=batch_size
    # num_classes=len(train_dataset.coco.cats)
    num_classes=max(train_dataset.coco.cats.keys())
    coco_classes=train_dataset.coco_classes
    # print("coco_classes:",train_dataset.coco.cats)

    # print("num_classes:",num_classes)

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    # 由于本项目图片视频统一采用1080P，所以不需要这个参数
    '''
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    '''

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    
    nw = min([os.cpu_count(), batch_size if batch_size > 4 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> instances_val2017.json
    # val_dataset = CocoDetection(data_root, "val", data_transform["val"])
    val_dataset =CocoDetection(data_root, TRAIN_DATASETS_SAVE_PATH+"/"+config.actionId,"val",data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    model = create_model(coco_classes, config.actionId)
    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []
    maxMap = 0.0001

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weightDecay)

    scaler = torch.amp.autocast_mode("cuda") if config.amp else None

    # print("Start training-----"+str(scaler))

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[10,20,50],
                                                        gamma=0.1)
    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if config.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(config.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.startEpoch = checkpoint['epoch'] + 1
        if config.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(config.startEpoch, config.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=10,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device,logId=config.trainLogId)
        if det_info!=None and seg_info!=None:
            txt = ""
            # write detection into txt
            with open(det_results_file, "a",encoding="utf-8") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # write seg into txt
            with open(seg_results_file, "a",encoding="utf-8") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")
            print("this model mAP值：------->",det_info[0])
            val_map.append(det_info[0])  # pascal mAP
            if det_info[0] > maxMap and epoch>0:
                print("save best model mAP值：------->",det_info[0])
                maxMap = det_info[0]
                # save weights
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'coco_classes': coco_classes
                }
                if config.amp:
                    save_files["scaler"] = scaler.state_dict()
                localSavePath=TRAIN_MODEL_SAVE_PATH+"/"+config.trainLogId+".pth"
                torch.save(save_files,localSavePath)
                osName=platform.system().lower()
                if(osName == "linux"):
                    # or osName=="darwin"  苹果   windows
                    # 将模型训练文件推送到推理服务器
                    time.sleep(1)
                    rsync_push(local_path = localSavePath, 
                                remote_path= TRAIN_MODEL_SAVE_PATH,
                                remoteServer=linuxServer(
                                    isNeedSudo=True, 
                                    ip=DEDUCTION_SERVER_IP,
                                    port=DEDUCTION_SERVER_SSH_PORT, 
                                    user=DEDUCTION_SERVER_SSH_USER,
                                    password=DEDUCTION_SERVER_SSH_PASSWORD),
                                local_password=TRAIN_SERVER_SSH_PASSWORD,
                                shell=True)
                
                # torch.save(save_files, TRAIN_MODEL_SAVE_PATH+"/"+config.trainLogId+"_"+str(epoch)+".pth")
            # 每一轮日志回传
            # 调用java接口回传训练日志
            threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_TRAIN_LOG_TO_JAVA, {
                "trainingRecordListId": config.trainLogId,
                "mAP": f'{det_info[0]:.6f}',
                "epoch":f"{epoch}",
                "logLv": "1",
                "logsContent": txt
            })).start()
        else:
            threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_TRAIN_LOG_TO_JAVA, {
                "trainingRecordListId": config.trainLogId,
                "mAP": '0',
                "epoch":f"{epoch}",
                "logLv": "1",
                "logsContent": "本轮训练失效"
            })).start()
        if redis and redis.get(f"{START_TRAINING_PX}{config.trainLogId}")=="end":
            print("训练任务终止结束了")
            break
    # 调用java接口回传训练结果
    threading.Thread(target=postJsonWithOutJwt, args=(JAVA_API_PATH + SEND_TRAIN_RESULT_TO_JAVA, {
        "trainingRecordListId": config.trainLogId,
        "mAP": f'{maxMap:.6f}'
    })).start()
    print("训练结束")
    '''
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)
    '''
'''
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='/data/coco2017', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=90, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.004, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练的batch size(如果内存/GPU显存充裕，建议设置更大)
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
'''
