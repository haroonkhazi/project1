import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('training.csv')
labels = train_data['label']
imgData = train_data.drop(['label'], axis=1).values



decision_tree_classifier = DecisionTreeClassifier(criterion='entropy',
                                max_depth=64, max_features=748,
                                min_samples_leaf=3, min_samples_split=3,
                                splitter='best')
decision_tree_classifier.fit(imgData[:14000], labels[:14000])
label_pred = decision_tree_classifier.predict(imgData[14000:])
print(accuracy_score(np.array(labels[14000:]), label_pred))