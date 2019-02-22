import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score

train = pd.read_csv("training.csv")
train.head()
lab_encoder = LabelEncoder()
label = train["label"]
label_encoded = lab_encoder.fit_transform(label)
label_encoded
df2 = train.drop(columns = ["label"])
x_train = df2.values
Y_train = np.argmax(label_1hot, axis=1)
label = label.values
clf = MultinomialNB()
clf.fit(x_train[0:19999],Y_train[0:19999])
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
compare = []
for i in range(20000,21000):
    result = clf.predict(x_train[i].reshape(1, 784))
    compare.append(result[0])
len(compare) 
accuracy_score(compare, label[20000:21000])