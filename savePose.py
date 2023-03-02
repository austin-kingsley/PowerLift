import cv2
import numpy as np
import os
import PoseModule as pm
from enum import Enum
detector = pm.poseDetector()

for exercise in ["good", "bad"]:
    directory = os.fsencode("C:\\Users\\austi\\Desktop\\3yp\\powerlifting\\squatform\\" + exercise)
    # directory = os.fsencode("C:\\Users\\austi\\Desktop\\opencv\\test")

    def is_black_image(image):
        return cv2.countNonZero(image[:,:,0]) == 0 and cv2.countNonZero(image[:,:,1]) == 0 and cv2.countNonZero(image[:,:,2]) == 0

    def save_pose(detector, source_dir, dest_dir):
        for file in os.listdir(source_dir):
            filename = os.path.basename(file)

            # if exercise == "deadlift":
            #     print("a")
            #     img = cv2.imread('C:\\Users\\austi\\Desktop\\opencv\\test\\download (1).jfif')
            #     detector = pm.poseDetector()
            #     img = detector.findPoseEmpty(img, True)
            #     while True:
            #         cv2.imshow("blah", img)
            #         cv2.waitKey()

            detector = pm.poseDetector()
            img = cv2.imread(os.path.join(source_dir, file))
            img = detector.image_resize(img, width = 300)
            img = detector.findPoseEmpty(img, True)
            if is_black_image(img):
                # cv2.imwrite(os.path.join(dest_dir, filename + ".png"), img)
                continue
            img = detector.cropToPose(img)
            cv2.imwrite(os.path.join(dest_dir, filename + "3.png"), img)
    

    save_pose(detector, 'C:\\Users\\austi\\Desktop\\3yp\\powerlifting\\squatform\\' + exercise, 'C:\\Users\\austi\\Desktop\\3yp\\powerlifting\\squatformpose\\' + exercise)
    # save_pose(detector, 'C:\\Users\\austi\\Desktop\\opencv\\test', 'C:\\Users\\austi\\Desktop\\opencv\\test')