from sklearn import svm, preprocessing
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
import numpy as np


'''
Classifier
'''

# load dataset from pickled numpy files
class_labels = np.load("class_labels.npy")
dataset = np.load("dataset.npy")

# normalize dataset
dataset = preprocessing.scale(dataset)

# initialize svc
# first try
# C = 1.0  # SVM regularization parameter
# train svc
# svc = svm.SVC(kernel='linear', C=C).fit(training_scaled, classes)
# with multiclass

X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset, class_labels, test_size=0.5)

clf = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(X_train, y_train)
# give
predict = clf.predict(X_test)

result_sum = (predict == y_test).sum()
accuracy = result_sum/2089
print(result_sum)
print(accuracy*100)

print(clf.score(X_test, y_test))
