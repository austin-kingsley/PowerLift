import pickle
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import PoseModule as pm
import cv2
detector = pm.poseDetector()

# open a file, where you stored the pickled data
file = open('sbd-model2.p', 'rb')

# dump information to that file
model = pickle.load(file)

# close the file
file.close()

input_dir = 'C:\\Users\\austi\\Desktop\\3yp\\pics\\deadlift'

# img = detector.findPoseEmpty(img, True)
# while True:
#     cv2.imshow("blah", img)
#     cv2.waitKey()

for file in os.listdir(input_dir):
    detector = pm.poseDetector()
    filename = os.path.basename(file)
    img = cv2.imread(os.path.join(input_dir, file))
    img = detector.image_resize(img, width = 300)
    

    detector.findPose(img)
    lm = detector.findPosition(img, False)

    if not lm:
        print("IDK - " + str(filename))
        continue
    
    lm = list(lm.values())
    lm = [[l[0], l[1]] for l in lm]
    lm = [item for sublist in lm for item in sublist]

    # a.reshape(1, -1)
    exercise = model.predict([lm])

    exercise = ["SQUAT", "BENCH", "DEADLIFT"][exercise[0]]

    # print(exercise)
    print(exercise + " - " + str(filename))