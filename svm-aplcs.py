# Applying SVM to different problems, Cancer Detection
# & Character Recognition.  

#=====================================================
# Author: Marjan Khamesian
# Date: August 2020
#=====================================================

# --------------------------------------
# Example(1) : Cancer Detection with SVM
# --------------------------------------

## Loading dataset
from sklearn import datasets

cancer = datasets.load_breast_cancer()
print(cancer.data[3])

## Exploring data
print(cancer.data.shape)
print(cancer.target)

## Splitting data
X = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=109)


##### Applying SVM ####
from sklearn import svm

## Creating a classifier
cls = svm.SVC(kernel="linear")

## Training the model
cls.fit(X_train,y_train)

## Prediction
pred = cls.predict(X_test)

## Evaluation
from sklearn import metrics

## Accuracy
print("Acuracy:", metrics.accuracy_score(y_test,y_pred=pred))

## Precision score
print("Precision:", metrics.precision_score(y_test,y_pred=pred))

## Recall score
print("Recall", metrics.recall_score(y_test,y_pred=pred))

print(metrics.classification_report(y_test, y_pred=pred))

############################################
# ------------------------------------------
# Example(2): Character Recognition with SVM
# ------------------------------------------

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

## Loading the dataset
letters = datasets.load_digits()

## Generating the classifier
clf = svm.SVC(gamma=0.001, C=100)

## Training the classifier
X,y = letters.data[:-10], letters.target[:-10]
clf.fit(X,y)

## Prediction 
print(clf.predict(letters.data[:-10]))
plt.imshow(letters.images[6], interpolation='nearest')
plt.show()
