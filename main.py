import pytube

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import os
import re

from matplotlib import patches
from tqdm import tqdm

from utils import *
from model import *
import argparse
import detection_and_tracking as det
from PIL import Image

# import libraries for videos
import pytube
from IPython.display import HTML


def draw_circle(event, x, y, flags, param):
    '''
    Callback used in cv2 to draw the circles and the lines  of the perimeter
    of the ROI using the mouse click and to take the relevant points useful for the perspective
    transformation.
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    '''
    global edit_frame, mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_points) < MAX_MOUSE_POINTS:
            cv2.circle(edit_frame, (x, y), 5, (0, 0, 255), -1)
            mouse_points.append([x, y])

            # draw line between points
            if len(mouse_points) >= 2 and len(mouse_points) <= MAX_MOUSE_POINTS:
                cv2.line(edit_frame,
                         (mouse_points[len(mouse_points) - 2][0],
                          mouse_points[len(mouse_points) - 2][1]),
                         (mouse_points[len(mouse_points) - 1][0],
                          mouse_points[len(mouse_points) - 1][1]),
                         (0, 255, 0), 2)
                if len(mouse_points) == MAX_MOUSE_POINTS:
                    cv2.line(edit_frame, (mouse_points[0][0], mouse_points[0][1]),
                             (mouse_points[len(mouse_points) - 1][0], mouse_points[len(mouse_points) - 1][1]),
                             (0, 255, 0), 2)
                    print(mouse_points)


if __name__ == "__main__":

    # DEFINISCO I PARAMETRI CON ARG PARSER
    parser = argparse.ArgumentParser(
        description="Tool to analyze videos and check if people respect social distancing.")
    parser.add_argument("video_file", type=str, help="video file to process. Could be .mp4")
    args = parser.parse_args()

    # RIPRENDO IL VIDEO E LO DIVIDO IN FRAME
    #video_file = args.video_file
    #first_frame, FPS, video_name = save_frames_from_video(video_file)
    #se non devo dividere in frames
    video_name = "sample1"
    file_frame = "234.png"
    first_frame = cv2.imread(FRAMES_FOLDER + video_name + "/" + file_frame)
    mouse_points = []
    width, height, _ = first_frame.shape
    print("width e height ", width, height)

    # PRENDO IL PRIMO FRAME PER TRACCIARE LA ROI
    # set window
    cv2.namedWindow("Trace points of ROI")
    cv2.setMouseCallback("Trace points of ROI", draw_circle)
    edit_frame = first_frame.copy()
    # visualize window and start to trace ROI
    # trace points must follow bottom-left, bottom-right, top-right, top-left order
    while (True):
        cv2.imshow("Trace points of ROI", edit_frame)
        key_pressed = cv2.waitKey(20)
        # press ESC to stop drawing
        if key_pressed == 27:
            exit(0)
        # clean image and restart to take mouse points
        # if 'r' button is pressed
        if key_pressed == 114:
            mouse_points = []
            edit_frame = first_frame.copy()

        if key_pressed == 32:
            edit_frame = first_frame.copy()
            break

    cv2.destroyWindow("Trace points of ROI")
    # end to trace ROI

    # CALCOLO LA MATRICE DI PERSPECTIVE
    matrix_transformation, frame_transformed = compute_perspective_transform(mouse_points, width, height, first_frame)
    bird_view_image = frame_transformed #np.zeros(frame_transformed.shape)

    # CARICO IL MODELLO DA UTILIZZARE
    #model, Tensor = det.load_darknet_model(det.CONFIG_PATH, det.WEIGHTS_PATH, det.IMG_SIZE)

    #detectron
    detectron_model = define_model()

    # PER OGNI FRAME CALCOLO LA DETECTION PER TROVARE I PUNTI IN BASSO DEI BBOXES
    # cap = cv2.VideoCapture(video_file)
    frame_files = os.listdir(FRAMES_FOLDER + video_name)
    frame_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # for file in tqdm(frame_files):

    # read
    #frame = Image.open(FRAMES_FOLDER + video_name + "/" + file_frame)
    #frame = cv2.imread(FRAMES_FOLDER + video_name + "/" + file_frame)
    # fid people
    #people_detected = det.detect_people(frame, model, Tensor, det.IMG_SIZE)
    #people_detected = detectron_model(frame)
    #print(people_detected)
    midpoints = find_people(FRAMES_FOLDER + video_name + "/",file_frame,detectron_model)
    print(midpoints)



    '''
    frame = np.array(frame)

    #plt.figure()
    #fig, ax = plt.subplots(1, figsize=(12, 9))
    #ax.imshow(frame)
    #pad_x = max(height - width, 0) * (det.IMG_SIZE / max(frame.shape))
    #pad_y = max(width - height, 0) * (det.IMG_SIZE / max(frame.shape))
    #unpad_h = det.IMG_SIZE - pad_y
    #unpad_w = det.IMG_SIZE - pad_x

    # find the central point of each person
    midpoints = []
    if people_detected is not None:
        #people_dict = det.process_people(people_detected, height, width, unpad_h, unpad_w, pad_x, pad_y)
        people_dict = det.process_people(people_detected, height, width)

    for key in people_dict.keys():
        top_left_bbox = people_dict[key]['top-left-bbox']
        box_w = people_dict[key]['box_w']
        box_h = people_dict[key]['box_h']
        bottom_right_bbox = people_dict[key]['bottom-right-bbox']
        midpoint = people_dict[key]['midpoint']
        midpoints.append(midpoint)

        # draw relative detections
        cv2.circle(first_frame, (midpoint[0],midpoint[1]), 3, color=det.PEOPLE_WARNING_COLOR)
        cv2.rectangle(first_frame,top_left_bbox,bottom_right_bbox,color=det.PEOPLE_NORMAL_COLOR)


    transformed_midpoints = compute_point_perspective_transformation(matrix_transformation,midpoints)
    # design midpoint on birdview image
    for point in transformed_midpoints:
        x1, y1 = point

        cv2.circle(bird_view_image,(x1,y1),3,color=det.PEOPLE_NORMAL_COLOR)

    cv2.imshow("frame",first_frame)
    cv2.waitKey(0)
    cv2.imshow("bird_view",bird_view_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    '''
    num_frame = 0
    # Process each frame, until end of video
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            #convert to PIL image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(frame)
            print("pil w, h " +str(pilimg.size[1]),str(pilimg.size[0]))


            #find people
            people_detected = det.detect_people(pilimg,model,Tensor,det.IMG_SIZE)
            frame = np.array(pilimg)
            #find the central point of each person
            #if people_detected is not None:


            cv2.imshow("frame",frame)
            num_frame+=1
            cv2.waitKey(int(FPS))
        else:
            print("end of the video file...")
            break
    cap.release()
    cv2.destroyAllWindows()


    # E TRASFORMARLI NELLA PERSPECTIVE
    # name_video = "sample1"
    # path_video = VIDEO_FOLDER + name_video + ".mp4"
    # frame_file = FRAMES_FOLDER + name_video + "/" + "1" + ".png"
    # first_frame, FPS = save_frames_from_video(path_video)

    
    cv2.namedWindow("bird")
    black_image = np.zeros(frame_transformed.shape)
    cv2.imshow("bird",black_image)
    cv2.waitKey(0)
    cv2.destroyWindow("bird")
    cv2.imwrite("bird_view.png",black_image)
    '''
