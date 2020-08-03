import os
from copy import deepcopy

import detectron2
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

from utils import mid_point, get_frames, FRAMES_FOLDER, compute_perspective_transform, \
    compute_perspective_unit_distances, return_people_ids, compute_point_perspective_transformation, compute_distances, \
    check_risks_people, COLOR_SAFE, COLOR_WARNING, COLOR_DANGEROUS

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


def perform_social_detection(video_name, points_ROI, points_distance, width, height, selected_model):
    output_folder = "out/"
    # create output folder of processed frames
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(output_folder + video_name):
        os.mkdir(output_folder + video_name)

    # get frames to be processed
    frames = get_frames(FRAMES_FOLDER + video_name)

    # first frame
    frame_file = FRAMES_FOLDER + video_name + "/" + frames[0]
    img = cv2.imread(frame_file)

    # get perspective transformation points of ROI and distance
    matrix_transformation, bird_eye_frame = compute_perspective_transform(points_ROI, width, height, img)
    distance_w, distance_h = compute_perspective_unit_distances(points_distance, matrix_transformation)

    # define model based on parameter 'selected model'
    if selected_model == 'yolo':
        print("find distances with YoloV3")
        yolov3_model_path = "./yolo.h5"
        model, custom_objects = YoloV3_model(yolov3_model_path)
    elif selected_model == 'fasterRCNN':
        print("find distances with fasterRCNN")
        model = faster_RCNN_model()

    # get info from bird eye frame
    bird_width, bird_height, _ = bird_eye_frame.shape
    diff_height_bg = bird_height - height
    width_bg = width

    # create background image for text
    background_img = np.zeros((diff_height_bg, width_bg, 3), dtype=np.uint8)
    background_img[:diff_height_bg, :width_bg] = (127, 127, 127)

    # process over the frames
    for f in tqdm(frames):
        # read the frame and create a copy of background image
        frame = cv2.imread(FRAMES_FOLDER + video_name + "/" + f)
        background_social_detector = deepcopy(background_img)
        # create bird-eye-view image
        bird_eye_view_img = np.zeros((bird_height, bird_width, 3))

        # choose the right predict based on 'selected_model' parameter
        if selected_model == 'yolo':
            bboxes, midpoints = find_people_YoloV3(frame, model, custom_objects)
        elif selected_model == 'fasterRCNN':
            bboxes, midpoints = find_people_fasterRCNN(frame, model)

        # return the indices of the people detected
        people_ids = return_people_ids(bboxes)

        # perform operations on frame if is detected at least 1 person
        if len(midpoints) > 0:
            # transform midpoints based on the matrix perspective transformation
            # calculate the distances on bird eye
            midpoints_transformed = compute_point_perspective_transformation(matrix_transformation, midpoints)
            dist_bird, dist_line = compute_distances(midpoints_transformed, distance_w, distance_h)

            # divide the people in the right sets based on the distance calculated
            set_safe_faster, set_warning_faster, set_dangerous_faster = check_risks_people(dist_bird, people_ids)

            # Draw the boxes on the frame based on the warning degree
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                if i in set_safe_faster:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_SAFE)
                elif i in set_warning_faster:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WARNING)
                elif i in set_dangerous_faster:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DANGEROUS)

                    # Draw circles of right color on bird eye image
            for i in range(len(midpoints_transformed)):
                x, y = midpoints_transformed[i][0], midpoints_transformed[i][1]
                if i in set_safe_faster:
                    cv2.circle(bird_eye_view_img, (x, y), 5, COLOR_SAFE, 5)
                elif i in set_warning_faster:
                    cv2.circle(bird_eye_view_img, (x, y), 5, COLOR_WARNING, 5)
                elif i in set_dangerous_faster:
                    cv2.circle(bird_eye_view_img, (x, y), 5, COLOR_DANGEROUS, 5)

                # set text to write on background image based on statistics
                text_number_people = "People detected: " + str(len(midpoints_transformed))
                text_safe = "Safe person: " + str((len(set_safe_faster) / len(midpoints_transformed)) * 100) + "%"
                text_warning = "Warning person: " + str(
                    (len(set_warning_faster) / len(midpoints_transformed)) * 100) + "%"
                text_dangerous = "Dangerous person: " + str(
                    (len(set_dangerous_faster) / len(midpoints_transformed)) * 100) + "%"
        else:
            # no people detected, write only 0 on the background image
            text_number_people = "People detected: 0"
            text_safe = "Safe person: 0.0%"
            text_warning = "Warning person: 0.0%"
            text_dangerous = "Dangerous person: 0.0%"

        # set text on image
        cv2.putText(background_social_detector, text_number_people, (12, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                    2, cv2.LINE_4)
        cv2.putText(background_social_detector, text_safe, (12, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SAFE, 2,
                    cv2.LINE_4)
        cv2.putText(background_social_detector, text_warning, (12, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WARNING,
                    2, cv2.LINE_4)
        cv2.putText(background_social_detector, text_dangerous, (12, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLOR_DANGEROUS, 2, cv2.LINE_4)

        # compose the image
        # numpy_vertical = np.vstack((frame, background_social_detector))
        numpy_vertical_concat = np.concatenate((frame, background_social_detector), axis=0)

        # numpy_horizontal = np.hstack((numpy_vertical_concat, bird_eye_view_img))
        numpy_horizontal_concat = np.concatenate((numpy_vertical_concat, bird_eye_view_img), axis=1)

        # write result of edit frame
        cv2.imwrite(output_folder + video_name + "/" + f, numpy_horizontal_concat)
