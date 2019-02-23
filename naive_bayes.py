import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score

train = pd.read_csv("training.csv")
lab_encoder = LabelEncoder()
label = train["label"]
label_encoded = lab_encoder.fit_transform(label)
label_encoded
df2 = train.drop(columns = ["label"])
x_train = df2.values
Y_train = np.argmax(label_1hot, axis=1)
label = label.values
clf = MultinomialNB()
clf.fit(x_train[0:14000],Y_train[0:14000])# 2/3 of training data
#Using a Mulinomial Naive Bayes
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
compare = []
for i in range(14001,21000): #testing on 1/3 of training data
    result = clf.predict(x_train[i].reshape(1, 784))
    compare.append(result[0])
accuracy_score(compare, label[14001:21000]) # comparing 1/3 label data with predicted lables
