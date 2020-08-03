import numpy as np
import cv2
import os
import re
import pytube

# DEFINE SOME CONSTANTS
VIDEO_FOLDER = "./video/"
FRAMES_FOLDER = "./frames/"
MAX_MOUSE_POINTS = 7
MAX_POINTS_ROI = 4


def download_from_youtube(video_url, folder_path, video_name=None):
    '''
    Method used to download video from Youtube
    :param video_url: the url of the video to download
    :param folder_path: the folder path where save the video
    :param video_name: the name to give to the save file
    :return:
    '''
    youtube = pytube.YouTube(video_url)
    video = youtube.streams.first()

    video.download(folder_path)  # path, where to video download.
    # #if u want to rename the video
    # if not video_name is None:
    #    os.rename(folder_path + video.title + ".mp4", video_name + ".mp4")
    #    return video_name
    return video.title


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
    if not os.path.isfile(video_path):
        IOError("File video doesn't exists!")
        return

    # check if exists frames dir, otherwise create it
    if not os.path.isdir('frames'):
        os.mkdir('frames')

    # take video name to rename save folder
    video_name = get_video_name(video_path)

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

    # read first frame
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


def get_frames(frames_dir):
    frames = os.listdir(frames_dir)
    frames.sort(key=lambda f: int(re.sub('\D', '', f)))
    return frames


def get_video_name(video_path):
    return video_path.split("/")[-1].split(".")[0]


def compute_perspective_transform(corner_points, width, height, image):
    ''' Compute the transformation matrix
	:param corner_points : 4 corner points selected from the image
	:param  height, width : size of the image
	return : transformation matrix and the transformed image
	'''
    # Create an array out of the 4 corner points
    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    # order is left-bottom, right-bottom, right-top, left-top
    img_params = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed


def compute_perspective_unit_distances(unit_points, matrix_transformation):
    '''
    Points must be: central, width, height order
    :param unit_points:
    :param matrix_transformation:
    :return:
    '''
    # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
    points_distance = np.float32(np.array([unit_points]))
    warped_pt = cv2.perspectiveTransform(points_distance, matrix_transformation)[0]

    # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
    # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
    # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
    # which we can use to calculate distance between two humans in transformed view or bird eye view
    distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
    distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
    return distance_w, distance_h


def compute_point_perspective_transformation(matrix, list_downoids):
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
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    return transformed_points_list


def mid_point(person):
    # get the coordinates
    x1, y1, x2, y2 = person

    # compute bottom center of bbox
    x_mid = int((x1 + x2) / 2)
    y_mid = int(y2)
    mid = (x_mid, y_mid)

    return mid


def write_results(file_txt, video_name, FPS, width, height, mouse_points):
    # create file .txt and write results
    f = open(file_txt, "w+")

    f.write("video name:" + video_name + '\n')
    f.write("FPS:" + str(FPS) + '\n')
    f.write("width:" + str(width) + '\n')
    f.write("height:" + str(height) + '\n')
    f.write("points of ROI:\n")
    for p in mouse_points[:4]:
        x, y = p
        f.write(str(x) + "," + str(y) + "\n")

    f.write("points of distance:\n")
    for p in mouse_points[4:7]:
        x, y = p
        f.write(str(x) + "," + str(y) + "\n")
    f.close()


def read_results(file_txt):
    b_take_ROI = False
    b_take_distance = False
    points_ROI = []
    points_distance = []
    video_name = ""
    FPS = 0.0
    width = 0
    height = 0

    with open(file_txt) as fr:
        for line in fr.readlines():
            if "video_name" in line:
                video_name = line.split(":")[1]
            elif "FPS:" in line:
                FPS = float(line.split(":")[1])
            elif "width:" in line:
                width = int(line.split(":")[1])
            elif "height:" in line:
                height = int(line.split(":")[1])
            elif "points of ROI:" in line:
                b_take_ROI = True
            elif "points of distance:" in line:
                b_take_ROI = False
                b_take_distance = True
            elif b_take_ROI:
                point = line.split(",")
                points_ROI.append([int(point[0]), int(point[1])])
            elif b_take_distance:
                point = line.split(",")
                points_distance.append([int(point[0]), int(point[1])])

        fr.close()
    return video_name, FPS, width, height, points_ROI, points_distance
