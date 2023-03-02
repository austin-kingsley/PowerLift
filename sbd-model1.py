import os
import pickle
import cv2

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import PoseModule


# prepare data
input_dir = 'C:\\Users\\austi\\Desktop\\3yp\\pics'
categories = ["squat", "bench", "deadlift"]
# print(os.getcwd())
data = []
labels = []
n = 0
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        print("pics\\" + category + "\\"+ file)
        detector = PoseModule.poseDetector()
        img = cv2.imread("pics\\" + category + "\\"+ file)
        img = detector.image_resize(img, width = 300)

        detector.findPose(img)
        lm = detector.findPosition(img, False)

        if not lm:
            continue

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

        data.append(a)
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)


# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

# pickle.dump(best_estimator, open('./sbd-model.p', 'wb'))
# pickle.dump(best_estimator, open('./sbd-model1.p', 'wb'))