import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("training.csv")

def plotNum(ind):
    plt.imshow(np.reshape(np.array(data.iloc[ind,1:]), (28, 28)), cmap="gray")


plt.figure()
for ii in range(1,17):
    plt.subplot(4,4,ii)
    plotNum(ii)
plt.show()