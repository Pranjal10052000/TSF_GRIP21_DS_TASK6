
print("SOLUTION TO TSF_GRIP21_DS_TASK6 BY PRANJAL KALEKAR")

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

#importing data.
iris = datasets.load_iris()
X = pd.DataFrame(iris.data , columns=iris.feature_names)
y = iris.target
print("Data: ")
print(X.head())

#classifying data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#lets define Dicision tree classifier
des_tree = DecisionTreeClassifier()

#fitting data into classifier
des_tree.fit(X_train,y_train)
print("model has been trained")

y_pred = des_tree.predict(X_test)

features = iris.feature_names
species = iris.target_names

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)
tree.plot_tree(des_tree, feature_names = features , class_names = species, filled = True);