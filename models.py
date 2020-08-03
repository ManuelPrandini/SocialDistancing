import detectron2
from detectron2.utils.logger import setup_logger

from utils import mid_point

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

# import libraries for Yolo
from imageai.Detection import ObjectDetection


def faster_RCNN_model():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cuda'

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.94  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


def YoloV3_model(yolov3_model_path):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()  # Se vuoi usare yolo tiny cambia il set model
    detector.setModelPath(yolov3_model_path)
    custom_objects = detector.CustomObjects(person=True)
    detector.loadModel()
    return detector, custom_objects


def find_people_fasterRCNN(frame, model):
    # img = cv2.imread(frame_file)
    outputs = model(frame)
    classes = outputs['instances'].pred_classes.cpu().numpy()
    bbox = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    ind = np.where(classes == 0)[0]
    people = bbox[ind]
    midpoints = [mid_point(person) for person in people]
    return people, midpoints


def find_people_YoloV3(frame, model, custom_objects=None):
    returned_image, detections = model.detectCustomObjectsFromImage(
        custom_objects=custom_objects,
        input_type="array", input_image=frame,
        output_type="array",
        minimum_percentage_probability=30
    )
    people = [x['box_points'] for x in detections]
    midpoints = [mid_point(person) for person in people]
    return people, midpoints
