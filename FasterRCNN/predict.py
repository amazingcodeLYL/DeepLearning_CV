import torch
from torchvision import transforms
from model.faster_rcnn import FasterRCNN
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from train_utils.draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2

def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    return model


# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# create model
model = create_model(num_classes=21)

# load train weights
train_weights = "./save_weights/fasterrcnn_resnet50_fpnmodel_39.pth"
model.load_state_dict(torch.load(train_weights)["model"])
model.to(device)

# read class_indict
category_index = {}
try:
    json_file = open('dataset_process/pascal_voc_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

# load image
original_img = Image.open("./test.jpg")

# from pil image to tensor, do not normalize image
data_transform = transforms.Compose([transforms.ToTensor()])
img = data_transform(original_img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

model.eval()
with torch.no_grad():
    predictions = model(img.to(device))[0]
    predict_boxes = predictions["boxes"].to("cpu").numpy()
    predict_classes = predictions["labels"].to("cpu").numpy()
    predict_scores = predictions["scores"].to("cpu").numpy()

    if len(predict_boxes) == 0:
        print("没有检测到任何目标!")

    draw_box(original_img,
             predict_boxes,
             predict_classes,
             predict_scores,
             category_index,
             thresh=0.5,
             line_thickness=5)

    plt.imshow(original_img)
    plt.savefig("img.jpg")
    plt.show()
