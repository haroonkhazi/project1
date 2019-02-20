import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("training.csv")
data2 = pd.read_csv("testing.csv")

X = data.iloc[:, 1:]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rfc = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features='auto', max_depth=50, bootstrap=False)
rfc.fit(X_train, y_train)
y_predict_rfc = rfc.predict(X_test)
print(y_predict_rfc)
acc_score = accuracy_score(y_test, y_predict_rfc)
print(acc_score)

X_testingdata = data2.iloc[:, 0:]
y_predict_rfc_testingdata = rfc.predict(X_testingdata)
y_predict_rfc_testingdata = pd.Series(y_predict_rfc_testingdata)
data2.insert(loc=0, column="label", value=y_predict_rfc_testingdata)
data2.to_csv("testingwithlabels", sep=',', encoding='utf-8')

data = pd.read_csv("testingwithlabels")
X = data.iloc[:, 1:]
y = data['label']
y_predict_rfc = rfc.predict(X)
print(y_predict_rfc)
acc_score = accuracy_score(y, y_predict_rfc)
print(acc_score)