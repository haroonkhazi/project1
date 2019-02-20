import csv
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv("testingdata16.csv")


def plotNum(ind):
    plt.imshow(np.reshape(np.array(data.iloc[ind, 0:]), (28, 28)), cmap="gray")


plt.figure()
for ii in range(1,17):
    plt.subplot(4,4,ii)
    plotNum(ii)
plt.show()

with open("testingdata16.csv", 'r') as csvinput:
    with open("test.csv", 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)
        


