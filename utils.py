import numpy as np
import cv2
import os
import re
import pytube

#DEFINE SOME CONSTANTS
VIDEO_FOLDER = "./video/"
FRAMES_FOLDER = "./frames/"
MAX_MOUSE_POINTS = 4

def download_from_youtube(video_url,folder_path,video_name = None):
    '''
    Method used to download video from Youtube
    :param video_url: the url of the video to download
    :param folder_path: the folder path where save the video
    :param video_name: the name to give to the save file
    :return:
    '''
    youtube = pytube.YouTube(video_url)
    video = youtube.streams.first()
    video.download(folder_path) # path, where to video download.
    # #if u want to rename the video
    if not video_name is None:
        os.rename(folder_path + video.title, video_name + ".mp4")


def get_frame_rate(video_capture):
    '''
    Method that return the FPS of a video
    :param video_capture: the Videocapture object of a video
    :return: the number of FPS
    '''
    FPS = video_capture.get(cv2.CAP_PROP_FPS)
    print("frame_rate: " + str(FPS))
    return FPS


def save_frames_from_video(video_path):
    '''
    Method that takes the path of a video, read the video
    and create all the frames of the video, saving them on
    the specific folder of the video inside the frames folder
    :param video_path: path of the input video
    :return: the first frame and the FPS, and the video name
    '''
    # check if exists frames dir, otherwise create it
    if not os.path.isdir('frames'):
        os.mkdir('frames')

    # take video name to rename save folder
    video_name = video_path.split("/")[-1].split(".")[0]

    # define where save the frames and create
    # the folder if not exists yet
    save_path_folder = FRAMES_FOLDER + video_name + "/"
    if not os.path.isdir(save_path_folder):
        os.mkdir(save_path_folder)

    print("Save frames from " + video_path + " ...")
    # capture video
    cap = cv2.VideoCapture(video_path)
    cnt = 0

    # check frame rate
    FPS = get_frame_rate(cap)

    # Check if video file is opened successfully
    if (cap.isOpened() == False):
        IOError("Error opening video stream or file")

    #read first frame
    ret, first_frame = cap.read()

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            # save each frame to folder
            cv2.imwrite(save_path_folder + str(cnt) + '.png', frame)
            cnt = cnt + 1
            if (cnt == 750):
                break

        # Break the loop
        else:
            print("Done! " + str(cnt) + " frames saved in" + save_path_folder)
            return first_frame, FPS, video_name


def compute_perspective_transform(corner_points,width,height,image):
	''' Compute the transformation matrix
	:param corner_points : 4 corner points selected from the image
	:param  height, width : size of the image
	return : transformation matrix and the transformed image
	'''
	# Create an array out of the 4 corner points
	corner_points_array = np.float32(corner_points)
	# Create an array with the parameters (the dimensions) required to build the matrix
    #order is left-bottom, right-bottom, right-top, left-top
	img_params = np.float32([[0,height],[width,height],[width,0],[0,0]])
	# Compute and return the transformation matrix
	matrix = cv2.getPerspectiveTransform(corner_points_array,img_params)
	img_transformed = cv2.warpPerspective(image,matrix,(width,height))
	return matrix,img_transformed


def compute_point_perspective_transformation(matrix,list_downoids):
	''' Apply the perspective transformation to every ground point which have been detected on the main frame.
	:param  matrix : the 3x3 matrix
	:param  list_downoids : list that contains the points to transform
	return : list containing all the new points
	'''
	# Compute the new coordinates of our points
	list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
	transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
	# Loop over the points and add them to the list that will be returned
	transformed_points_list = list()
	for i in range(0,transformed_points.shape[0]):
		transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
	return transformed_points_list