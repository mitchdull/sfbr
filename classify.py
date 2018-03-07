'''
Classify single images (for fun)
Input:
Image of building in Dataset
Output: label
'''
import argparse

import numpy as np
from cv2 import imread

from sklearn import svm, preprocessing
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import data_prep as dp


if __name__ == '__main__':
    file_name = "sheffield_test.jpg"
    kernel_size = 5
    theta_list = [0,45,90,135,180,225,270,315]

    class_labels = np.load("class_labels.npy")
    dataset = np.load("dataset.npy")
    dataset_256 = np.load("dataset_256.npy")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to image to classify")
    args = parser.parse_args()

    if args.image:
        file_name = args.image

    img = imread(file_name,0)
    feature_maps = []
    for theta in theta_list:
        feature_maps.append(dp.steerableGaussian(img,theta,kernel_size))
        feature_maps.append(dp.steerableHilbert(img,theta,kernel_size))

    lda_input = np.array([], dtype=np.uint8)
    for feature_map in feature_maps:
        pooled = dp.imageMaxPool(feature_map)
        lda_input = np.append(lda_input,pooled)

    lda_input = np.resize(lda_input,(1,256))

    lda = LinearDiscriminantAnalysis(n_components=39)
    reduced = lda.fit(dataset_256, class_labels).transform(lda_input)

    scaler = preprocessing.StandardScaler().fit(dataset)
    scaler.transform(reduced)

    clf = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(dataset, class_labels)

    print(clf.predict(reduced))
