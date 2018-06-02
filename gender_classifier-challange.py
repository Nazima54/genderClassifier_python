# Build a Decision tree using Scikit-learn
# Gender Classification Challange
# Use 5 different Classifiers and find the best
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

#(height, weight, shoesize)
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# DECISION TREE CLASSIFIER
# "DT_clf" variable here will store the decision tree classifier
clf1 = tree.DecisionTreeClassifier()   # call the tree classifier
DT_clf = clf1.fit(X,Y) # Fit method trains the decision tree on our dataset.

# KNearestNeighbor CLASSIFIER
# knn will store the KnearestNeighbor classifier when KNN = 5
clf2 = KNeighborsClassifier(n_neighbors=5)
knn_clf = clf2.fit(X,Y)

# Support Vector Machine Classifier
clf3 = svm.SVC(probability=True)
svm_clf = clf3.fit(X,Y)

# MPL Classifier
clf4 = MLPClassifier(learning_rate = 'constant', learning_rate_init = 0.001,)
MLP_clf = clf4.fit(X,Y)

# GaussiamProcess Classsifier
clf5 = GaussianProcessClassifier()
gaussian_clf = clf5.fit(X,Y)

# Test the input
test = [[160,60,38]]

# "Prediction" variable  will store the predicted value

# Storing Results
DT_prediction = DT_clf.predict(test)
knn_prediction = knn_clf.predict(test)
svm_prediction = svm_clf.predict(test)
MLP_prediction = MLP_clf.predict(test)
gaussian_prediction = gaussian_clf.predict(test)

# Storing Probablities
prob_DT = DT_clf.predict_proba(test)
prob_knn = knn_clf.predict_proba(test)
prob_svm = svm_clf.predict_proba(test)
prob_MLP = MLP_clf.predict_proba(test)
prob_gaussian = gaussian_clf.predict_proba(test)

print("DT Classifier test data {} is predicted as {} with probability of {}".format(test,DT_prediction,prob_DT))
print("KNN Classifier test data {} is predicted as {} with probability of {}".format(test,knn_prediction,prob_knn))
print("SVM Classifier test data {} is predicted as {} with probability of {}".format(test,svm_prediction,prob_svm))
print("MLP Classifier test data {} is predicted as {} with probability of {}".format(test,MLP_prediction,prob_MLP))
print("Gaussian Classifier test data {} is predicted as {} with probability of {}".format(test,gaussian_prediction,prob_gaussian))










