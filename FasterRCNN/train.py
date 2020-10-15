import torch
import torchvision
from train_utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from model.faster_rcnn import FasterRCNN
from model.utils.rpn_function import AnchorsGenerator
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from model.faster_rcnn import FasterRCNN, FastRCNNPredictor
from backbone.mobilenetv2_model import MobileNetV2
from dataset_process.voc_dataset import VOC2012DataSet
from train_utils import train_eval_utils as utils, transforms
import os
from  backbone.vgg_model import  vgg
import time
import datetime
def select_model(num_classes):
    if args.model=="fasterrcnn_resnet50_fpn":
        backbone = resnet50_fpn_backbone()
        model = FasterRCNN(backbone=backbone, num_classes=91)
        # # 载入预训练模型权重
        # # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
        weights_dict = torch.load("./model/model_path/fasterrcnn_resnet50_fpn_coco.pth")
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        # get number of input features for the classifier得到分类器的输入特征数量
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one用一个新的取代预训练头
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    elif args.model=="MobileNetV2":
        backbone = MobileNetV2(weights_path="./model/model_path/mobilenet_v2.pth").features
        backbone.out_channels = 1280
    elif args.model=="vgg16":
        backbone = vgg(model_name="vgg16", weights_path="./model/model_path/vgg16-397923af.pth").features
        backbone.out_channels = 512
    elif args.model=="vgg11":
        backbone = torchvision.models.vgg11(pretrained=True).features
        backbone.out_channels = 512
    #生成anchors  size：框大小 aspect_ratios:长宽比
    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    #
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model



def load_data(args):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path
    assert os.path.exists(os.path.join(VOC_root, "VOCdevkit")), "not found VOCdevkit in path:'{}'".format(VOC_root)

    # load  data set
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data_set)
        test_sampler = torch.utils.data.SequentialSampler(val_data_set)

    if args.aspect_ratio_group_factor >= 0:
        # 统计所有图像比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_data_set, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_sampler=train_batch_sampler,
                                                    num_workers=args.workers,
                                                    collate_fn=utils.collate_fn)

    # load validation data set

    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=1,
                                                      sampler=test_sampler,
                                                      num_workers=args.workers,
                                                      collate_fn=utils.collate_fn)
    return train_data_loader,val_data_set_loader,train_sampler,train_batch_sampler

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    train_data_loader, val_data_set_loader, train_sampler, train_batch_sampler=load_data(args)

    # create model num_classes equal background + 20 classes
    model = select_model(num_classes=21)
    model.to(device)
    model_without_ddp = model
    #DataParallel是单进程，多线程的，并且只能在单台机器上运行，DistributedDataParallel而是多进程的，并且可以在单机和多机训练中使用。
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    train_loss = []
    learning_rate = []
    val_mAP = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone and train 5 epochs                   #
    #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for param in model.backbone.parameters():
        param.requires_grad = False

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    num_epochs = 5
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch,  print_freq=args.print_freq,
                              train_loss=train_loss, train_lr=learning_rate)

        # evaluate on the test dataset
        utils.evaluate(model, val_data_set_loader, device=device, mAP_list=val_mAP)

    torch.save(model.state_dict(), "./save_weights/pretrain.pth")
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # #  second unfrozen backbone and train all network     #
    # #  解冻前置特征提取网络权重（backbone），接着训练整个网络权重  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # # 冻结backbone部分底层权重
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    # learning rate scheduler

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_step_size,
                                                   gamma=args.lr_gamma)
    #在分布式训练时一般使用MultiStepLR
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    #若之前有训练好的模型 ，加载模型权重
    if args.resume != "":
        checkpoint = torch.load(args.resume,map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(args.start_epoch))

    if args.test_only:
        utils.evaluate(model, val_data_set_loader, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch,args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, print_freq=args.print_freq,
                              train_loss=train_loss, train_lr=learning_rate)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        utils.evaluate(model, val_data_set_loader, device=device, mAP_list=val_mAP)
        #epoch>10时模型基本收敛了
        if args.output_dir and epoch>10:
            # 只在主节点上执行保存权重操作
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, args.model+'model_{}.pth'.format(epoch)))

    utils.evaluate(model, val_data_set_loader, device=device, mAP_list=val_mAP)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from train_utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_mAP) != 0:
        from train_utils.plot_curve import plot_map
        plot_map(val_mAP)


if __name__ == "__main__":
    version = torch.version.__version__[:5]  # example: 1.6.0
    # 因为使用的官方的混合精度训练是1.6.0后才支持的，所以必须大于等于1.6.0
    if version < "1.6.0":
        raise EnvironmentError("pytorch version must be 1.6.0 or above")

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='/home/dell/', help='dataset')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--dataset', default='/home/dell/', help='dataset')
    parser.add_argument('--model', default='MobileNetV2', help='model fasterrcnn_resnet50_fpn MobileNetV2 vgg16 vgg11 ')
    parser.add_argument('--num_classes', default='21', help='num_classes')
    parser.add_argument('--min-size', default='600', help='image resize')
    parser.add_argument('--max-size', default='1000', help='image resize')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=17, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=5, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.33, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--env', default='faster-rcnn', help='visdom env')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
