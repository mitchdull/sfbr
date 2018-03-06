from sklearn import svm, preprocessing
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
import numpy as np


'''
Classification Test
'''

# load dataset from pickled numpy files
class_labels = np.load("class_labels.npy")
dataset = np.load("dataset.npy")

# normalize dataset
dataset = preprocessing.scale(dataset)

# split train/test data 50/50
X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset, class_labels, test_size=0.5)

#train classifier
clf = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(X_train, y_train)

# generate predictions for test data
predict = clf.predict(X_test)

# percent correctly predicted
result_sum = float((predict == y_test).sum())
print((result_sum/len(X_test)))

# Accuracy on train/test split
print(clf.score(X_test, y_test))
