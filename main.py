import pytube

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import os
import re
from utils import *

#import libraries for videos
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
    global frame, mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_points) < MAX_MOUSE_POINTS:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            mouse_points.append([x,y])

            # draw line between points
            if len(mouse_points) >=2 and len(mouse_points) <= MAX_MOUSE_POINTS :
                cv2.line(frame,
                        ( mouse_points[len(mouse_points)-2][0],
                          mouse_points[len(mouse_points)-2][1]),
                        ( mouse_points[len(mouse_points)-1][0],
                          mouse_points[len(mouse_points)-1][1]),
                        (0,255,0),2)
                if len(mouse_points) == MAX_MOUSE_POINTS:
                    cv2.line(frame,(mouse_points[0][0],mouse_points[0][1]),
                             (mouse_points[len(mouse_points) - 1][0],mouse_points[len(mouse_points) - 1][1]) ,
                             (0, 255, 0),2)
                    print(mouse_points)


if __name__ == "__main__":
    #FPS SAMPLE 2 --> 30
    #FPS SAMPLE_1 --> 24
    name_video = "sample1"
    path_video = VIDEO_FOLDER + name_video + ".mp4"
    frame_file = FRAMES_FOLDER + name_video + "/" + "1" + ".png"
    #first_frame, FPS = save_frames_from_video(path_video)

    mouse_points = []
    #read image
    frame = cv2.imread(frame_file)
    width, height, _ = frame.shape

    #set window
    cv2.namedWindow("Trace points of ROI")
    cv2.setMouseCallback("Trace points of ROI", draw_circle)

    #visualize window
    while(True):
        cv2.imshow("Trace points of ROI",frame)
        key_pressed = cv2.waitKey(20)
        #press ESC to stop drawing
        if  key_pressed == 27:
            exit(0)
        #clean image and restart to take mouse points
        # if r button is pressed
        if  key_pressed == 114:
            mouse_points = []
            frame = cv2.imread(frame_file)
        if  key_pressed == 32:
            frame = cv2.imread(frame_file)
            break

    cv2.destroyWindow("Trace points of ROI")
    #end part of ROI

    matrix_transformation, frame_transformed = compute_perspective_transform(mouse_points,width,height,frame)
    print("matrix ",matrix_transformation)
    cv2.namedWindow("bird")
    black_image = np.zeros(frame_transformed.shape)
    cv2.imshow("bird",black_image)
    cv2.waitKey(0)
    cv2.destroyWindow("bird")
    cv2.imwrite("bird_view.png",black_image)

