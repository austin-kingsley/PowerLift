import pickle
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import PoseModule as pm
import cv2
# detector = pm.poseDetector()

# open a file, where you stored the pickled data
file = open('sbd-modell.p', 'rb')

# dump information to that file
model = pickle.load(file)

# close the file
file.close()

input_dir = 'C:\\Users\\austi\\Desktop\\3yp\\pics\\deadlift'

# img = detector.findPoseEmpty(img, True)
# while True:
#     cv2.imshow("blah", img)
#     cv2.waitKey()

def is_black_image(image):
    return cv2.countNonZero(image[:,:,0]) == 0 and cv2.countNonZero(image[:,:,1]) == 0 and cv2.countNonZero(image[:,:,2]) == 0

for file in os.listdir(input_dir):
    filename = os.path.basename(file)
    img = cv2.imread(os.path.join(input_dir, file))

    img = resize(img, (15, 15)).flatten()
    img = np.array(img)
    img = img.reshape(-1, 675)

    exercise = model.predict(img)

    exercise = ["SQUAT", "BENCH", "DEADLIFT"][exercise[0]]

    # print(exercise)
    print(exercise + " - " + str(filename))