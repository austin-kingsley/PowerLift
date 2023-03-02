import torch
import cv2
import numpy as np
from skimage.transform import resize

def findPlate(img, plateDetectionModel):
    #Model
    # plateDetectionModel = torch.hub.load('./yolo/yolov5-master', 'custom', path='./yolo/yolov5-master/best.pt', source='local')  # local repo
    
    # Inference
    results = plateDetectionModel(img)  # includes NMS
    # print(results)

    boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)

    if boxes.empty:
        return None, False

    bbox = (int(boxes["xmin"][0]), int(boxes["ymin"][0]), int(boxes["xmax"][0]-boxes["xmin"][0]), int(boxes["ymax"][0]-boxes["ymin"][0]))
    return bbox, True

def findPlateAlt(img, x, y):
    circleCentre = []
    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (17,17), 0)
    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                                param1=100, param2= 30, minRadius=0, maxRadius=100)

    def withinBox(chosen, bbox):
        centreX = chosen[0]
        centreY = chosen[1]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        return x < centreX < x + w and y < centreY < y + h

    circleCentre = (x, y)
    bbox = (circleCentre[0]-100, circleCentre[1]-100, 200, 200)

    if circles is None:
        print("Cannot find bar")
        return bbox

    circles = np.uint16(np.around(circles))
    chosen = None

    for c in circles[0, :]:
        if withinBox(c, bbox):
            chosen = c
            break

    if chosen is None:
        return bbox
    
    bbox = list(bbox)
    bbox[0] = chosen[0] - bbox[2]*0.5
    bbox[1] = chosen[1] - bbox[3]*0.5
    return tuple(bbox)

def predict1(img, model, detector):
    img = detector.image_resize(img, width = 300)

    img = detector.findPoseEmpty(img, True)

    # while True:
    #     cv2.imshow("A", img)
    #     cv2.waitKey(1)

    def is_black_image(image):
        return cv2.countNonZero(image[:,:,0]) == 0 and cv2.countNonZero(image[:,:,1]) == 0 and cv2.countNonZero(image[:,:,2]) == 0

    if is_black_image(img):
        print("ERROR: no person")
        return "SQUAT"

    img = detector.cropToPose(img)

    img = resize(img, (15, 15)).flatten()
    img = np.array(img)
    img = img.reshape(-1, 675)

    exercise = model.predict(img)
    exercise = ["SQUAT", "BENCH", "DEADLIFT"][exercise[0]]
    return exercise


def predict(img, model, detector):
    img = detector.image_resize(img, width = 300)
    

    detector.findPose(img)
    lm = detector.findPosition(img, False)
    if not lm:
        print("ERROR: no person")
        return "SQUAT"
    
    mostVisibleKnee = 25 if lm[25][3] > lm[26][3] else 26

    kneeJoint = mostVisibleKnee
    hipJoint = mostVisibleKnee-2
    ankleJoint = mostVisibleKnee+2
    shoulderJoint = mostVisibleKnee-14
    elbowJoint = mostVisibleKnee-12

    kneeAngle = detector.findAngle(img, hipJoint, kneeJoint, ankleJoint, draw=False)
    kneeAngle = 360 - kneeAngle if 360 - kneeAngle > 360 else kneeAngle

    hipAngle = detector.findAngle(img, shoulderJoint, hipJoint, kneeJoint, draw=False)
    hipAngle = 360 - hipAngle if 360 - hipAngle > 360 else hipAngle

    armAngle = detector.findAngle(img, elbowJoint, shoulderJoint, hipJoint, draw=False)
    armAngle = 360 - armAngle if 360 - armAngle > 360 else armAngle

    a = (kneeAngle, hipAngle, armAngle)

    exercise = model.predict([a])
    
    exercise = ["SQUAT", "BENCH", "DEADLIFT"][exercise[0]]
    return exercise