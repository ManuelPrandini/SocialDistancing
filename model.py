import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def define_model():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor

# define a function which return the bottom center of every bbox
def mid_point(img, person, idx):
    # get the coordinates
    x1, y1, x2, y2 = person[idx]
    #_ = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # compute bottom center of bbox
    x_mid = int((x1 + x2) / 2)
    y_mid = int(y2)
    mid = (x_mid, y_mid)

    #_ = cv2.circle(img, mid, 5, (0, 0, 255), -1)
    #cv2.putText(img, str(idx), mid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return mid

def find_people(video_path,name,model):

  img = cv2.imread(video_path + name)
  outputs = model(img)
  classes = outputs['instances'].pred_classes.cpu().numpy()
  bbox = outputs['instances'].pred_boxes.tensor.cpu().numpy()
  ind = np.where(classes==0)[0]
  person = bbox[ind]
  midpoints = [mid_point(img,person,i) for i in range(len(person))]
  num = len(midpoints)
  return midpoints

if __name__ == '__main__':
    pass
