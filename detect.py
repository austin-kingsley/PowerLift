import torch
import cv2
import numpy as np
import random

# detect the weight plate using YOLOv5 model
def findPlate(img, plateDetectionModel):
    # run object detection model on input image
    results = plateDetectionModel(img)

    # get bounding boxes for all detected weight plates
    boxes = results.pandas().xyxy[0]

    # iterate through list of bounding boxes to find the closest plate, i.e. the one with the max height
    closestPlate = max(range(len(boxes)), key=lambda i: boxes["ymax"][i]-boxes["ymin"][i]) if len(boxes) else None

    # a less elegant solution
    # maxWidth = 0
    # closestPlate = 0
    # for i in range(len(boxes)):
    #     m = boxes["xmax"][i]-boxes["xmin"][i]
    #     if m > maxWidth:
    #         maxWidth = m
    #         closestPlate = i

    # if no weight plates were detected return None and False
    if closestPlate is None:
        return None, False

    # extract coordinates of closest bounding box and return as a tuple
    bbox = (int(boxes["xmin"][closestPlate]), int(boxes["ymin"][closestPlate]), int(boxes["xmax"][closestPlate]-boxes["xmin"][closestPlate]), int(boxes["ymax"][closestPlate]-boxes["ymin"][closestPlate]))
    return bbox, True

# redundant measure to detect the weight plate using HoughCircles
def findPlateAlt(img, x, y):
    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    blurFrame = cv2.GaussianBlur(grayFrame, (17,17), 0) # apply Gaussian blur to reduce noise
    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2= 30, minRadius=0, maxRadius=100) # detect circles

    # check if a detected circle lies within a bounding box
    def withinBox(chosen, bbox):
        centreX = chosen[0]
        centreY = chosen[1]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        return x < centreX < x + w and y < centreY < y + h

    # define bounding box based on input coords x and y
    bbox = (x-100, y-100, 200, 200)

    # if no circles detected, return original bounding box
    if circles is None:
        return bbox 
    
    # sort detected circles by radius (descending) and convert to int
    circles = sorted(circles, key=lambda circle: -circle[2])
    circles = np.uint16(np.around(circles))

    # loop through detected circles and check if any lie within bounding box
    chosen = None
    for c in circles[0, :]:
        if withinBox(c, bbox):
            chosen = c
            break

    # if a circle is chosen then return a bounding box around it, else return original bounding box
    return (x-100, y-100, chosen[0]-100, chosen[1]-100) if chosen is not None else bbox

# depricated exercise classifier
def predict1(img, model, detector):
    img = detector.image_resize(img, width = 300) # resize image to fixed width
    img = detector.findPoseEmpty(img, True) # find the pose

    # check if image is completely black
    def is_black_image(image):
        return cv2.countNonZero(image[:,:,0]) == 0 and cv2.countNonZero(image[:,:,1]) == 0 and cv2.countNonZero(image[:,:,2]) == 0
    if is_black_image(img):
        print("ERROR: no person")
        return "SQUAT"

    # crop image and reshape
    img = detector.cropToPose(img)
    img = np.array(img)
    img = img.reshape(-1, 675)

    # predict exercise using SVM
    exercise = model.predict(img)
    exercise = ["Squat", "Bench", "Deadlift"][exercise[0]]
    return exercise

# classify exercise
def predict(img, model, detector):
    img = detector.image_resize(img, width = 300) # resize image to fixed width
    detector.findPose(img, True) # find the pose
    lm = detector.findPosition(img, False) # extract landmarks
    if not lm: # if no pose found, return random exercise
        print("ERROR: no person")
        return random.choice(["Squat", "Bench", "Deadlift"])
    
    # find most visible knee
    mostVisibleKnee = 25 if lm[25][3] > lm[26][3] else 26
    
    # find corresponding related joints, based on BlazePose definition
    kneeJoint = mostVisibleKnee
    hipJoint = mostVisibleKnee-2
    ankleJoint = mostVisibleKnee+2
    shoulderJoint = mostVisibleKnee-14
    elbowJoint = mostVisibleKnee-12
    wristJoint = mostVisibleKnee-10

    # find and normalise knee angle
    kneeAngle = detector.findAngle(img, hipJoint, kneeJoint, ankleJoint, draw=False)
    kneeAngle = 360 - kneeAngle if kneeAngle > 180 else kneeAngle

    # find and normalise hip angle
    hipAngle = detector.findAngle(img, shoulderJoint, hipJoint, kneeJoint, draw=False)
    hipAngle = 360 - hipAngle if hipAngle > 180 else hipAngle

    # find and normalise shoulder angle
    shoulderAngle = detector.findAngle(img, elbowJoint, shoulderJoint, hipJoint, draw=False)
    shoulderAngle = 360 - shoulderAngle if shoulderAngle > 180 else shoulderAngle

    # find and normalise arm angle
    armAngle = detector.findAngle(img, wristJoint, elbowJoint, shoulderJoint, draw=False)
    armAngle = 360 - armAngle if armAngle > 180 else armAngle

    yDistanceBetweenWristAndShoulder = lm[wristJoint][1] - lm[shoulderJoint][1] # calculate y-distance between the wrist and shoulder
    xDistanceBetweenShoulderAndHip = abs(lm[shoulderJoint][0] - lm[hipJoint][0]) # calculate x-distances between shoulder and hip
    yDistanceBetweenShoulderAndHip = lm[shoulderJoint][1] - lm[hipJoint][1] # calculate y-distances between shoulder and hip

    # create feature vector
    a = (kneeAngle, hipAngle, shoulderAngle, armAngle, yDistanceBetweenWristAndShoulder, xDistanceBetweenShoulderAndHip, yDistanceBetweenShoulderAndHip)

    # predict exercise using SVM
    exercise = model.predict([a])
    exercise = ["Squat", "Bench", "Deadlift"][exercise[0]]
    return exercise