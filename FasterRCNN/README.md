# FasterRCNN


## Prepare data
1. Download the training, validation, test data and VOCdevkit

   ```Bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```Bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```Bash
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

## Environment configuration
```sh
torch==1.6.0
scipy==1.2.1
torchvision==0.7.0
matplotlib==3.1.3
numpy==1.18.1
lxml==4.5.0
Pillow==7.2.0
pycocotools==2.0.2
```


## Train

```sh
python train.py -b 8 --epochs 16 --model fasterrcnn_resnet50_fpn --print-freq 20
```
## Distributed training

```sh
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --model fasterrcnn_resnet50_fpn --epochs 26
  --lr-steps 16 22 --aspect-ratio-group-factor 3
```

<img src="https://github.com/amazingcodeLYL/DeepLearning_CV/blob/main/FasterRCNN/picture/loss_and_lr.png" width="50%" height="50%" >
<img src="https://github.com/amazingcodeLYL/DeepLearning_CV/blob/main/FasterRCNN/picture/mAP.png" width="50%" height="50%"  >


## Result
|Implementation|mAP|
| :-----| :-----|
| [Origin paper](https://arxiv.org/pdf/1506.01497.pdf) | 0.699|


| Backbone|  EPochs |lr |batch_size|mem/GPU| pre-training | FPS |AP IOU=0.50|weights|
| :-----| :-----| :-----| ----: | ----: |  :----: | :----: |:----: |:----: |
| Resnet50_fpn | 16| 0.005 |8| 6848 | True |19.8 |0.828| [Resnet50_fpn](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth)|
| Resnet50_fpn | 40| 0.005 |8| 6848 | True |20.2 |0.832||
| Mobilenetv2 | 16| 0.005 |8| 1488 | True |38 |0.676|[Mobilenetv2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)|
| vgg16|16|0.005|8|4965|True|20.4|0.546||
| vgg16|16|0.001|8|4964|True|21.4|0.354||
| vgg11 | 16| 0.005 |8| 6848 | True |29.5 |0.236||



| Backbone| AP IOU=0.50:0.95 |IOU=0.50 |IOU=0.75 |IOU=0.50:0.95&area=small |IOU=0.50:0.95&area=medium |IOU=0.50:0.95&area=large| maxDets |
| :-----| :-----| :-----| ----: |  :----: | :----: |:----: | :----: |
| Resnet50_fpn | 0.521| 0.828 | 0.582 | 0.263 |0.397 |0.569|100|
| Mobilenetv2 | 0.295| 0.626 | 0.230 | 0.055 |0.180 |0.344|100|
| vgg16|0.241|0.546|0.165|0.085|0.175|0.277|100|
| vgg11 | 0.088| 0.236 | 0.039 |0.005 |0.047 |0.110|100|

<img src="https://github.com/amazingcodeLYL/DeepLearning_CV/blob/main/FasterRCNN/picture/img.jpg" width="50%" height="50%" style="float:right;" >
