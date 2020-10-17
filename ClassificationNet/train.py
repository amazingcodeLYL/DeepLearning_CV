import datetime
import os
import time
from torchsummary import summary
import torch
import torch.utils.data
import torch.nn as nn
import torchvision
from torchvision import transforms

import utils
try:
    from apex import amp
except ImportError:
    amp = None


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, dataset_len,print_freq, apex=False):
    model.train()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    # metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    running_loss=0.0
    running_corrects=0

    # for image, target in metric_logger.log_every(data_loader, print_freq, header):
    for step,data in enumerate(data_loader, start=0):
        start_time = time.time()
        image, target =data
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        _,preds=torch.max(output,1)
        running_loss+=loss.item()*image.shape[0]
        running_corrects+=torch.eq(preds,target).sum()

        # print train process
        rate = (step + 1) / len(data_loader)
        step = step + 1
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")

        # acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        # batch_size = image.shape[0]
        # metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
    train_loss=running_loss/dataset_len
    train_acc=running_corrects.double()*100/dataset_len
    # print("epoch_loss:",epoch_loss)
    print()
    print("train_acc: ",train_acc.item())

def evaluate(model,criterion, data_loader,dataset_test_len,device):
    model.eval()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # save_path="./resNet34.pth"
    with torch.no_grad():
        running_loss=0.0
        runnning_correct=0
        # for image, target in metric_logger.log_every(data_loader, print_freq, header):
        for data in data_loader:
            image,target=data
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            # acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            # batch_size = image.shape[0]

            _,pred=torch.max(output,1)
            runnning_correct+=torch.eq(pred,target).sum()


            # metric_logger.update(loss=loss.item())
            # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        test_acc=runnning_correct.double()*100/dataset_test_len
        print("test_acc: ",test_acc.item())
        return test_acc.item()
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

    # print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    # return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


# class DealDataset(Dataset):
#     """
#         下载数据、初始化数据，都可以在这里完成
#     """
#
#     def __init__(self):
#         xy = np.loadtxt('../dataSet/diabetes.csv.gz', delimiter=',', dtype=np.float32)  # 使用numpy读取数据
#         self.x_data = torch.from_numpy(xy[:, 0:-1])
#         self.y_data = torch.from_numpy(xy[:, [-1]])
#         self.len = xy.shape[0]
#
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
#
#     def __len__(self):
#         return self.len

# def load_data(traindir, valdir, cache_dataset, distributed):
#     # Data loading code
#     print("Loading data")
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#
#     print("Loading training data")
#     st = time.time()
#     cache_path = _get_cache_path(traindir)
#     if cache_dataset and os.path.exists(cache_path):
#         # Attention, as the transforms are also cached!
#         print("Loading dataset_train from {}".format(cache_path))
#         dataset, _ = torch.load(cache_path)
#     else:
#         dataset = torchvision.datasets.ImageFolder(
#             traindir,
#             transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ]))
#         if cache_dataset:
#             print("Saving dataset_train to {}".format(cache_path))
#             utils.mkdir(os.path.dirname(cache_path))
#             utils.save_on_master((dataset, traindir), cache_path)
#     print("Took", time.time() - st)
#
#     print("Loading validation data")
#     cache_path = _get_cache_path(valdir)
#     if cache_dataset and os.path.exists(cache_path):
#         # Attention, as the transforms are also cached!
#         print("Loading dataset_test from {}".format(cache_path))
#         dataset_test, _ = torch.load(cache_path)
#     else:
#         dataset_test = torchvision.datasets.ImageFolder(
#             valdir,
#             transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 normalize,
#             ]))
#         if cache_dataset:
#             print("Saving dataset_test to {}".format(cache_path))
#             utils.mkdir(os.path.dirname(cache_path))
#             utils.save_on_master((dataset_test, valdir), cache_path)
#
#     print("Creating data loaders")
#     if distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
#     else:
#         train_sampler = torch.utils.data.RandomSampler(dataset)
#         test_sampler = torch.utils.data.SequentialSampler(dataset_test)
#
#     return dataset, dataset_test, train_sampler, test_sampler



def load_data(distributed):
    #transform data
    #resnet mobilenet
    # transforms_train = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    # transform_test = transforms.Compose([transforms.Resize(256),
    #                            transforms.CenterCrop(224),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # #vggnet  alexnet  googlenet
    transforms_train=transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test=transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Data loading code
    print("Loading training data")
    st = time.time()
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms_train, download=True)
    print("Took", time.time() - st)
    dataset_len=len(dataset)
    print("Loading validation data")
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    dataset_test_len=len(dataset_test)

    # Distributed
    if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # Creating data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    return data_loader,data_loader_test,dataset_len,dataset_test_len,train_sampler,test_sampler

def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    # train_dir = os.path.join(args.data_path, 'train')
    # val_dir = os.path.join(args.data_path, 'val')
    # dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
    #                                                                args.cache_dataset, args.distributed)
    data_loader, data_loader_test,dataset_len,dataset_test_len,train_sampler,test_sampler=load_data(args.distributed)

    #迁移学习
    # model=torchvision.models.resnet50(pretrained=True).to(device)
    # for param in model.parameters():
    #     param.requires_grad=False
    # model.fc=nn.Sequential(nn.Linear(2048,128),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(128,10)).to(device)
    # optimizer=torch.optim.Adam(model.fc.parameters())
    # print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    model.to(device)

    #模型结构
    # model_summary=summary(model,(3,224,224))
    # print(model_summary)


    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    print("\rStart training")
    start_time = time.time()
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        print("epoch: ",[epoch])
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, dataset_len,args.print_freq, args.apex)

        lr_scheduler.step()

        test_acc=evaluate(model,criterion, data_loader_test, dataset_test_len,device)
        if args.output_dir and test_acc>best_acc:
            best_acc = test_acc
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            for root, dirs, files in os.walk(args.output_dir):
                if len(files) != 0:
                    for name in files:
                        os.remove(os.path.join(root, name))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'Best_Model{}.pth'.format(epoch)))
            # utils.save_on_master(
            #     checkpoint,
            #     os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', help='dataset')
    parser.add_argument('--model', default='vgg16', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=500, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
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
        action="store_true",#store_true就代表着一旦有这个参数，做出动作“将其值标为True”
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

