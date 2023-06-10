import cv2
import mediapipe as mp
import time
import math
import numpy as np
 
 
class poseDetector():
 
    def __init__(self, mode=False, complexity=1, smooth=True, segmentation = False, smooth_segmentation = True ,detectionCon=0.5, trackCon=0.5):
        
        # self.static_image_mode=True
        self.mode = mode
        self.complexity = complexity
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)
 
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPoseEmpty(self, img, draw=True):
        (h, w) = img.shape[:2]
        blank = np.zeros((h,w,3), dtype=np.uint8)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(blank, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return blank

    def cropToPose(self, img):
        y_nonzero, x_nonzero, _ = np.nonzero(img)
        return img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
 
    def findPosition(self, img, draw=True):
        self.lmList = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy, cz, viz = int(lm.x * w), int(lm.y * h), lm.z, lm.visibility
                # self.lmList.append([id, cx, cy, cz, viz])
                self.lmList.update({id: (cx, cy, cz, viz)})
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def hitDepth(self, img, draw=True):
        kneeY = self.lmList[26][2:][0]
        # print("Knee height: ")
        # print(kneeY)
        hipY = self.lmList[24][2:][0]
        # print("Hip height: ")
        # print(hipY)

        if draw:
            cv2.putText(img, str(hipY), (50, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(img, str(kneeY), (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if hipY < kneeY:
            return 0
        return 1

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized
 
    def findAngle(self, img, p1, p2, p3, draw=False):
 
        # Get the landmarks
        x1, y1 = self.lmList[p1][:2]
        x2, y2 = self.lmList[p2][:2]
        x3, y3 = self.lmList[p3][:2]
 
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
 
        # print(angle)
 
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
    
    def findDistance(self, p1, p2):
 
        # Get the landmarks
        x1, y1 = self.lmList[p1][:2]
        x2, y2 = self.lmList[p2][:2]
        
        return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

 
def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
 
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
 
if __name__ == "__main__":
    main()