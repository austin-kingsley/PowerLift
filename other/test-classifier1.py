import pickle
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import PoseModule as pm
import cv2
detector = pm.poseDetector()

# open a file, where you stored the pickled data
file = open('sbd-model1.p', 'rb')

# dump information to that file
model = pickle.load(file)

# close the file
file.close()

n, correct = 0, 0

test = ["SQUAT", "BENCH", "DEADLIFT"][0]

input_dir = 'C:\\Users\\austi\\Desktop\\3yp\\pics\\' + "deadlift-old"

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

    n += 1

    mostVisibleKnee = 25 if lm[25][3] > lm[26][3] else 26

    kneeJoint = mostVisibleKnee
    hipJoint = mostVisibleKnee-2
    ankleJoint = mostVisibleKnee+2
    shoulderJoint = mostVisibleKnee-14
    elbowJoint = mostVisibleKnee-12

    kneeAngle = detector.findAngle(img, hipJoint, kneeJoint, ankleJoint, draw=False)
    kneeAngle = 360 - kneeAngle if kneeAngle > 360 else kneeAngle

    hipAngle = detector.findAngle(img, shoulderJoint, hipJoint, kneeJoint, draw=False)
    hipAngle = 360 - hipAngle if hipAngle > 360 else hipAngle

    armAngle = detector.findAngle(img, elbowJoint, shoulderJoint, hipJoint, draw=False)
    armAngle = 360 - armAngle if armAngle > 360 else armAngle

    a = (kneeAngle, hipAngle, armAngle)


    # a.reshape(1, -1)
    exercise = model.predict([a])

    exercise = ["SQUAT", "BENCH", "DEADLIFT"][exercise[0]]

    if exercise == test:
        correct += 1

    # print(exercise)

    # if file == "download (5).jfif":
    #     while True:
    #         cv2.imshow("A", img)
    #         cv2.waitKey(1)


    print(exercise + " - " + str(filename))

print(correct/n * 100)