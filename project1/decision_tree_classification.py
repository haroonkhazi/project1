import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('training.csv')

labels = train_data['label']
imgData = train_data.drop(['label'], axis=1).values
tree_clf = DecisionTreeClassifier(max_depth=5, max_features=748)
tree_clf.fit(imgData[:14000], labels[:14000])

label_pred = tree_clf.predict(imgData[14000:])


np.random.seed(1)

param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 8, 16, 32, 64, 128], 'max_features': [748],
             'min_samples_leaf': [2, 3, 10], 'min_samples_split':[2, 3, 10], 'splitter': ['best', 'random']}

decision_tree_classifier = DecisionTreeClassifier()

grid_search = GridSearchCV(decision_tree_classifier, param_grid, cv = 3, scoring='accuracy')

grid_search.fit(imgData, labels)

print(grid_search.best_params_)
print(grid_search.best_score_)