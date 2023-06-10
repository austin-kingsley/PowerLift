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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# prepare data
input_dir = 'C:\\Users\\austi\\Desktop\\3yp\\pics'
categories = ["squat", "bench", "deadlift"]
# print(os.getcwd())
data = []
labels = []
n = 0

detector = PoseModule.poseDetector()

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        print("pics\\" + category + "\\"+ file)
        # detector = PoseModule.poseDetector()
        img = cv2.imread("pics\\" + category + "\\"+ file)
        img = detector.image_resize(img, width = 300)

        detector.findPose(img)
        lm = detector.findPosition(img, False)

        if not lm:
            print("idk ^")
            continue

        mostVisibleKnee = 25 if lm[25][3] > lm[26][3] else 26
        
        kneeJoint = mostVisibleKnee
        hipJoint = mostVisibleKnee-2
        ankleJoint = mostVisibleKnee+2
        shoulderJoint = mostVisibleKnee-14
        elbowJoint = mostVisibleKnee-12
        wristJoint = mostVisibleKnee-10

        kneeAngle = detector.findAngle(img, hipJoint, kneeJoint, ankleJoint, draw=False)
        kneeAngle = 360 - kneeAngle if kneeAngle > 180 else kneeAngle

        hipAngle = detector.findAngle(img, shoulderJoint, hipJoint, kneeJoint, draw=False)
        hipAngle = 360 - hipAngle if hipAngle > 180 else hipAngle

        shoulderAngle = detector.findAngle(img, elbowJoint, shoulderJoint, hipJoint, draw=False)
        shoulderAngle = 360 - shoulderAngle if shoulderAngle > 180 else shoulderAngle

        armAngle = detector.findAngle(img, wristJoint, elbowJoint, shoulderJoint, draw=False)
        armAngle = 360 - armAngle if armAngle > 180 else armAngle

        yDistanceBetweenWristAndShoulder = lm[wristJoint][1] - lm[shoulderJoint][1]
        xDistanceBetweenShoulderAndHip = abs(lm[shoulderJoint][0] - lm[hipJoint][0])
        yDistanceBetweenShoulderAndHip = lm[shoulderJoint][1] - lm[hipJoint][1]

        # a = (kneeAngle, hipAngle, shoulderAngle, armAngle)

        # a = (yDistanceBetweenWristAndShoulder, xDistanceBetweenShoulderAndHip, yDistanceBetweenShoulderAndHip)

        # a = (yDistanceBetweenWristAndShoulder, xDistanceBetweenShoulderAndHip, yDistanceBetweenShoulderAndHip)



        a = (kneeAngle, hipAngle, shoulderAngle, armAngle, yDistanceBetweenWristAndShoulder, xDistanceBetweenShoulderAndHip, yDistanceBetweenShoulderAndHip)

        data.append(a)
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)


# reduce data to 2 dimensions
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.02, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.1, 0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_train)

score = accuracy_score(y_prediction, y_train)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./sbd-model1-best.p', 'wb'))

# plot decision boundary
plot_decision_regions(data_reduced, labels, clf=best_estimator, legend=2)

handles, labels = plt.gca().get_legend_handles_labels()
labels = ['squat', 'bench', 'deadlift']
plt.legend(handles, labels, loc = 'upper left')

plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.savefig('decision_boundary1111.png')
plt.show()