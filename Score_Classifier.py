# Import modules
from collections import Counter
import os
import cv2
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


#############################################################
# Functions
#############################################################
# Balances dataset to minimize bias towards digits that occur more often
def balance_dataset(X, y):
    X_new = X[0:1]
    y_new = y[0:1]
    count = Counter(digit_labels)
    min_sample_size = min(count.values())
    labels = count.keys()
    for label in labels:
        location = np.where(y == label)[0]
        random_values = np.random.randint(0, len(location), min_sample_size)
        include_sample = location[random_values]
        X_new = np.append(X_new, X[include_sample], axis=0)
        y_new = np.append(y_new, y[include_sample])
    X_new = X_new[1:]
    y_new = y_new[1:]
    return X_new, y_new


#############################################################
# Program logic
# Program logic
#############################################################
# Create training data matrix and labels based on file names
os.chdir("TRAINING_DATA_PATH")
samples_list = os.listdir()
digit_matrix = []
digit_labels = []

for samples in samples_list:
    displayed_numbers = samples.split("_")[1].split(".")[0]

    # Load sample
    im = cv2.imread(samples)

    # Convert to grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Create thresholds
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    cont = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Create rectangles for contours
    rects = [cv2.boundingRect(c) for c in cont]
    im_gray = cv2.bitwise_not(im_gray)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    im_th = cv2.bitwise_not(im_th)

    if len(rects) == len(displayed_numbers):
        for i, rect in enumerate(rects):
            # Get label
            number_label = displayed_numbers[len(displayed_numbers) - 1 - i]

            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            number = im_th[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            number = cv2.resize(number, (28, 28), interpolation=cv2.INTER_AREA)
            number = cv2.dilate(number, (3, 3))
            number = number.flatten().reshape(1, -1)
            digit_matrix.append(list(number))
            digit_labels.append(list(number_label))

digit_matrix = np.array(digit_matrix)
digit_matrix = digit_matrix.reshape(digit_matrix.shape[0], -1)
digit_matrix.shape

digit_labels = np.array(digit_labels)
digit_labels = digit_labels.reshape(-1)
digit_labels.shape

print("Count of digits in dataset", Counter(digit_labels))

# Train a neural network and SVM
scaler = StandardScaler()
scaler.fit(digit_matrix)
scale_digit_matrix = scaler.transform(digit_matrix)

# Get balanced data set
X_new, y_new = balance_dataset(scale_digit_matrix.copy(), digit_labels.copy())

# Fit SVM
svm_param_grid = {'C': [1, 10, 50, 100, 1000]}
clf_balanced = GridSearchCV(estimator=LinearSVC(), param_grid=svm_param_grid)
clf_balanced.fit(X_new, y_new)
digit_predict_svm_balanced = clf_balanced.predict(X_new)
print(accuracy_score(y_new, digit_predict_svm_balanced))
os.chdir("MODEL_SAVE_PATH")
joblib.dump(clf_balanced, "SVM_number_recognition_balanced")
joblib.dump(scaler, "digit_scaler")

# Fit NN
nn = MLPClassifier()
nn_param_grid = {
    'hidden_layer_sizes': [(100, 1), (100, 50), (100, 50, 25), (500, 300, 100, 25), (600, 300, 200, 100, 25)],
    'alpha': [0.1, 0.01, 0.001, 0.0001]}
nn_balanced = GridSearchCV(estimator=nn, param_grid=nn_param_grid, cv=2)
digit_predict_nn_balanced = nn_balanced.predict(X_new)
print(accuracy_score(y_new, digit_predict_nn_balanced))
os.chdir("MODEL_SAVE_PATH")
joblib.dump(nn_balanced, "NN_number_recognition_balanced")
