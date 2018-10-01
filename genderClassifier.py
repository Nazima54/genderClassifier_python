# Build a Decision tree using Scikit-learn
# Gender Classification Challange
from sklearn import tree
#(height, weight, shoesize)
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# "clf" variable here will store the decision tree classifier
clf = tree.DecisionTreeClassifier()   # call the tree classifier

clf = clf.fit(X,Y) # Fit method trains the decision tree on our dataset.


# "Prediction" variable  will store the predicted value
prediction = clf.predict([[160,60,38]])

# print the prediction value
print(prediction)

