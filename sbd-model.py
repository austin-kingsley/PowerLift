import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import PoseModule as pm
import cv2


# prepare data
input_dir = 'C:\\Users\\austi\\Desktop\\3yp\\pics'
categories = ["squat", "bench", "deadlift"]
# print(os.getcwd())
data = []
labels = []
n = 0

def is_black_image(image):
    return cv2.countNonZero(image[:,:,0]) == 0 and cv2.countNonZero(image[:,:,1]) == 0 and cv2.countNonZero(image[:,:,2]) == 0

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        print(file)
        
        detector = pm.poseDetector()

        img = cv2.imread("pics\\" + category + "\\"+ file)

        img = detector.image_resize(img, width = 300)

        img = detector.findPoseEmpty(img, True)
        if is_black_image(img):
            print("err")
            continue

        img = detector.cropToPose(img)

        
        img = resize(img, (15, 15))

        # img = resize(img, (15, 15)).flatten()
        img = np.array(img)

        print(img)

        data.append(img.flatten())
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

# pickle.dump(best_estimator, open('./sbd-modella.p', 'wb'))