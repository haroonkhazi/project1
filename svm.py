import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib
from sklearn.metrics import accuracy_score
%matplotlib inline
train = pd.read_csv("training.csv")
lab_encoder = LabelEncoder()
label = train["label"]
label_encoded = lab_encoder.fit_transform(label)
label_encoded
df2 = train.drop(columns = ["label"])
x_train = df2.values
x_test = x_train[15000:21000] # testing on 1/3 of data
x_train =x_train[0:14999] # training on 2/3 of data 
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)
model=svm.SVC()
model.fit(x_train,label_encoded[0:14999])
compare = []
for i in range(0,6000):
    result = model.predict(x_test[i].reshape(1, 784))
    compare.append(result[0]) 
accuracy_score(compare, label[15000:21000])