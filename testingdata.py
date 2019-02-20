import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("testingwlabels_knc.csv")


def plotNum(ind):
    plt.imshow(np.reshape(np.array(data.iloc[ind, 2:]), (28, 28)), cmap="gray")

y = data['label']

plt.figure()
for ii in range(1,17):
    plt.subplot(4,4,ii,title="{}".format(y[ii+16]))
    plotNum(ii+16)
plt.show()
del data

