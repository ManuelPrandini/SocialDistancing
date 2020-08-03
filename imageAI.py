from imageai.Detection import ObjectDetection

detector = ObjectDetection()

yolov3_model_path = "yolo-tiny.h5"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

input_path = "frames/sample1/234.png"
output_path_yoloV3 = "output1.jpg"

# Detection with yoloV3 weights

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(yolov3_model_path)
detector.loadModel()

detection = detector.detectObjectsFromImage(input_image = input_path, output_image_path = output_path_yoloV3)

img1=mpimg.imread(output_path_yoloV3)
imgplot1 = plt.imshow(img1)
plt.show()

for x in detection:
    print(x["name"], " : ", x["percentage_probability"], " : ", x["box_points"])