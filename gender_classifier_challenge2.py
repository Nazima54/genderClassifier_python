# Build a Decision tree using Scikit-learn
# Gender Classification Challange
# Use 5 different Classifiers and find the best
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np

#(height, weight, shoesize)
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# DECLARE THE CLASSIFIER
clf_tree = tree.DecisionTreeClassifier()
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_svm = SVC(probability=True)
clf_MLP = MLPClassifier(learning_rate = 'constant', learning_rate_init = 0.001,)
clf_gaussian = GaussianProcessClassifier()
clf_perceptron = Perceptron()

# Training the model
clf_tree.fit(X,Y)
clf_knn.fit(X,Y)
clf_svm.fit(X,Y)
clf_MLP.fit(X,Y)
clf_gaussian.fit(X,Y)
clf_perceptron.fit(X,Y)

# Testing model
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y,pred_tree) * 100
print('Accuracy for decision tree is {}'.format(acc_tree))


pred_KNN = clf_knn.predict(X)
acc_KNN = accuracy_score(Y,pred_KNN) * 100
print('Accuracy for KNN is {}'.format(acc_KNN))

pred_SVM = clf_svm.predict(X)
acc_SVM = accuracy_score(Y,pred_SVM) * 100
print('Accuracy for SVM is {}'.format(acc_SVM))

pred_MLP = clf_MLP.predict(X)
acc_MLP = accuracy_score(Y,pred_MLP) * 100
print('Accuracy for MLP is {}'.format(acc_MLP))

pred_gaussian = clf_gaussian.predict(X)
acc_gaussian = accuracy_score(Y,pred_gaussian) * 100
print('Accuracy for gaussian is {}'.format(acc_gaussian))

pred_perceptron = clf_perceptron.predict(X)
acc_perceptron = accuracy_score(Y,pred_perceptron) * 100
print('Accuracy for perceptron is {}'.format(acc_perceptron))

# The best classifier is
index = np.argmax((acc_SVM,acc_perceptron,acc_KNN,acc_MLP,acc_gaussian))
classifiers = {0 : 'SVM',1 : 'Perceptron',3 : 'KNN',4 : 'MLP',5 : 'Gaussian'}
print('The gender classifier is {}'.format(classifiers[index]))






