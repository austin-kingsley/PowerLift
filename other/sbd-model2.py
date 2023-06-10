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
        # img = detector.findPoseEmpty(img, True)

        detector.findPose(img)
        lm = detector.findPosition(img, False)

        if not lm:
            continue

        lm = list(lm.values())
        lm = [[l[0], l[1]] for l in lm]
        lm = [item for sublist in lm for item in sublist]

        print(lm)

        data.append(lm)
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
pickle.dump(best_estimator, open('./sbd-model2.p', 'wb'))