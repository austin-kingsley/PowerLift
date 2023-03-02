import cv2
import os
import PoseModule as pm
detector = pm.poseDetector()


points = []

def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def get_image_coordinates(input_dir):
    
    # loop through all images in the input directory
    for file in os.listdir(input_dir):
        print(points)
        # read the image
        img = cv2.imread(os.path.join(input_dir, file))

        img = detector.image_resize(img, width = 300)
        
        # create a window to display the image
        cv2.imshow('Image', img)

        cv2.setMouseCallback("Image", mousePoints)
        
        # wait for the user to click somewhere in the window
        cv2.waitKey(0)
        
        # save the coordinate where the user clicked
        # x,y = cv2.getMousePos()
        # coordinates.append((x,y))
        
        # destroy the window
        cv2.destroyAllWindows()
    
    # return the list of coordinates


input_dir = 'C:\\Users\\austi\\Desktop\\opencv\\side\\combined'

get_image_coordinates(input_dir)

print(points)

input_dir = 'C:\\Users\\austi\\Desktop\\opencv\\side'

with open(os.path.join(input_dir, "coords.txt"), 'w') as f:
    f.write(str(points))
    f.write(len(points))