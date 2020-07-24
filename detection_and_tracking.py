import matplotlib

from pytorch_objectdetecttrack.models import *
from pytorch_objectdetecttrack.utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

from pytorch_objectdetecttrack.utils import utils as ext_utils
##LINK DOVE PRENDERE SPUNTO
# https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98

EXT_LIBRARY = "pytorch_objectdetecttrack/"
config_path=  EXT_LIBRARY + 'config/yolov3.cfg'
weights_path= EXT_LIBRARY +'config/yolov3.weights'
class_path= EXT_LIBRARY +'config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
PEOPLE_COLOR = (0,1,0)

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
#model.cuda()
model.eval()
classes = ext_utils.load_classes(class_path)
#Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
                                       transforms.Pad((max(int((imh-imw)/2),0),
                                                       max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
                                                       max(int((imw-imh)/2),0)), (128,128,128)),
                                       transforms.ToTensor(),
                                       ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = ext_utils.non_max_suppression(detections, 80,conf_thres, nms_thres)
    #filter and return only people detection
    print(detections)
    if detections[0] is not None:
        return detections[0][detections[0][:,-1] == 0]
    else:
        return None

if __name__ == '__main__':
    # load image and get detections
    img_path = "frames/sample1/160.png"
    prev_time = time.time()
    img = Image.open(img_path)
    img2 = Image.open("bird_view.png")
    detections = detect_image(img)
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print('Inference Time: %s' % (inference_time))  # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    img = np.array(img)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        # browse detections and draw bounding boxes
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            y2 = ((y2 - pad_y // 2) / unpad_h) * img.shape[0]
            x2 = ((x2 - pad_x // 2) / unpad_w) * img.shape[1]
            center = (x1 + x2) / 2


            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                     linewidth=2, edgecolor=PEOPLE_COLOR, facecolor='none')
            ax.add_patch(bbox)
            midpoint = patches.Circle((center,y2),3,color="red")
            ax.add_patch(midpoint)
            plt.text(x1, y1, s=classes[int(cls_pred)],
                     color='white', verticalalignment='top')
    plt.text(20,img.shape[0],s="People detected : "+str(len(detections)),color="black",verticalalignment='top',fontsize=15)
    plt.axis('off')
    # save image
    plt.savefig("out.png",
                bbox_inches='tight', pad_inches=0.0)
    plt.show()