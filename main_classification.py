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
print(y_predict_rfc[0:16])
acc_score = accuracy_score(y_test, y_predict_rfc)
print(acc_score)

knc = KNeighborsClassifier(weights='distance', n_neighbors=5, 
        leaf_size=71, algorithm='ball_tree')
knc.fit(X_train, y_train)
y_predict_knc = knc.predict(X_test)
print(y_predict_knc[0:16])
acc_score = accuracy_score(y_test, y_predict_knc)
print(acc_score)
del data

X_testingdata = data2.iloc[:, 0:]
y_predict_rfc_testingdata = rfc.predict(X_testingdata)
print(y_predict_rfc_testingdata[0:16])
y_predict_rfc_testingdata = pd.Series(y_predict_rfc_testingdata)
rfc_testingdata = pd.read_csv("testing.csv")
rfc_testingdata.insert(loc=0, column="label", value=y_predict_rfc_testingdata)
rfc_testingdata.to_csv("testingwlabels_rfc.csv", sep=',', encoding='utf-8')
del rfc_testingdata

y_predict_knc_testingdata = knc.predict(X_testingdata)
print(y_predict_knc_testingdata[0:16])
y_predict_knc_testingdata = pd.Series(y_predict_knc_testingdata)
knc_testingdata = pd.read_csv("testing.csv")
knc_testingdata.insert(loc=0, column="label", value=y_predict_knc_testingdata)
knc_testingdata.to_csv("testingwlabels_knc.csv", sep=',', encoding='utf-8')
del knc_testingdata
acc_score = accuracy_score(y_predict_knc_testingdata, y_predict_rfc_testingdata)



"""
rfc_tl_data = pd.read_csv("testingwlabels_rfc.csv")
X_rfc_tl = rfc_tl_data.iloc[:, 2:]
y_rfc_tl = rfc_tl_data['label']
y_predict_rfc = rfc.predict(X_rfc_tl)
print(y_predict_rfc)
acc_score = accuracy_score(y_rfc_tl, y_predict_rfc)
print(acc_score)

knc_tl_data = pd.read_csv("testingwlabels_knc.csv")
X_knc_tl = knc_tl_data.iloc[:, 2:]
y_knc_tl = knc_tl_data['label']
y_predict_knc = knc.predict(X_knc_tl)
print(y_predict_knc)
acc_score = accuracy_score(y_knc_tl, y_predict_knc)
print(acc_score)
"""




