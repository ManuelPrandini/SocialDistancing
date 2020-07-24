# import the necessary packages
from tkinter import *
from PIL import Image, ImageDraw
from PIL import ImageTk
from tkinter import filedialog
import cv2
import numpy as np

MAX_CONT = 7

def perspective_transform():
    global img_width, img_height
    global cv2_image
    # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
    # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
    # This bird eye view then has the property property that points are distributed uniformally horizontally and 
    # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
    # equally distributed, which was not case for normal view.
    src = np.float32(np.array(points[:4]))
    dst = np.float32([[0, img_height], [img_width, img_height], [img_width, 0], [0, 0]])
    prespective_transform = cv2.getPerspectiveTransform(src, dst)

    # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
    pts = np.float32(np.array([points[4:7]]))
    warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
            
    # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
    # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
    # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
    # which we can use to calculate distance between two humans in transformed view or bird eye view
    distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
    distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
    pnts = np.array(points[:4], np.int32)
    cv2.polylines(cv2_image, [pnts], True, (70, 70, 70), thickness=2)

#function to be called when mouse is clicked
def printcoords(event):
    global img
    global cont
    global points
    global canvas
    global message
    global img_width
    global img_height

    radius = 5
    color = "red" if cont < 4 else "blue"
    #outputting x and y coords to console
    if cont < MAX_CONT:
        if event.x > img_width or event.y > img_height:
            print("fuori")

        else: 
            x,y = event.x,event.y
            points.append((x,y))
            cont+=1
            #part of ROI
            if cont == 2:
                canvas.create_line(points[0][0],points[0][1],points[1][0],points[1][1], fill="green")
            if cont == 3:
                canvas.create_line(points[1][0],points[1][1],points[2][0],points[2][1], fill="green")
            if cont == 4:
                canvas.create_line(points[2][0],points[2][1],points[3][0],points[3][1], fill="green")
                canvas.create_line(points[0][0],points[0][1],points[3][0],points[3][1], fill="green")
            #end part of ROI

            #create circle
            x1, y1 = (event.x - radius), (event.y - radius)
            x2, y2 = (event.x + radius), (event.y + radius)
            canvas.create_oval(x1,y1,x2,y2, fill = color, outline =color)
    else:
        print(points)
        points = []
        canvas.create_image(0,0,image=img,anchor="nw")
        cont = 0

if __name__ == "__main__":
    root = Tk() 
    root.title("Social Distancing")
    cont = 0
    points = []
    #setting up a tkinter canvas with scrollbars
        #adding the image
    File = filedialog.askopenfilename(parent=root)
    cv2_image = cv2.imread(File)
    img = ImageTk.PhotoImage(Image.open(File))
    #draw = ImageDraw.Draw(img)
    

    img_width = img.width()
    img_height = img.height()

    root.geometry(str(img_width)+'x'+str(img_height))

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    canvas = Canvas(frame, bd=0)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    frame.pack(fill=BOTH,expand=1)

    panel = Label(frame)

    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    

    #mouseclick event
    canvas.bind("<Button 1>",printcoords)
    
    
    root.mainloop()