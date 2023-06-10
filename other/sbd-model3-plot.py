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


detector = pm.poseDetector()

def is_black_image(image):
    return cv2.countNonZero(image[:,:,0]) == 0 and cv2.countNonZero(image[:,:,1]) == 0 and cv2.countNonZero(image[:,:,2]) == 0

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        print(file)

        img = cv2.imread("pics\\" + category + "\\"+ file)

        img = detector.image_resize(img, width = 300)

        img = detector.findPoseEmpty(img, True)
        if is_black_image(img):
            print("idk ^")
            continue

        img = detector.cropToPose(img)

        
        img = resize(img, (15, 15))

        img = np.array(img)

        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# reduce data to 2 dimensions
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data_reduced, labels, test_size=0.05, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_train)

score = accuracy_score(y_prediction, y_train)

print('{}% of samples were correctly classified'.format(str(score * 100)))

# plot decision boundary
plot_decision_regions(data_reduced, labels, clf=best_estimator, legend=3)


handles, labels = plt.gca().get_legend_handles_labels()
labels = ['squat', 'bench', 'deadlift']
plt.legend(handles, labels, loc='lower right')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('decision_boundary3.png')
plt.show()