import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier

train_data = pd.read_csv('training.csv')

# For convolutional neural networks, the labels need to be one hot-encoded
# The pixel values for the black-and-white images range from 0 to 255
# so the values need to be scaled between 0 and 1 as floats.
# Additionally, the Conv2D layer requires a 3-D image as input because it deals with color images
# The images in this dataset are black-and-white 2-D images, but can be converted to a pseudo 3-D 
# image by adding a third dimension of 1
labelsData = train_data['label']
onehotData = keras.utils.to_categorical(labelsData, 10)
# Convert pandas data frame to numpy array
imgData = train_data.drop(['label'], axis=1).values
# Reshape into a 21000x28x28x1 matrix
imgData = imgData.reshape(21000, 28, 28, 1)
# Rescale pixel values
imgData = imgData / 255.0
# Sample image
plt.imshow(imgData[0].reshape(28, 28), cmap='gray')


smpSize = (28, 28, 1)

def create_model(Conv2D_size, kernel_size_conv, activation_func, pool_size_conv):
    model = Sequential()
    model.add(Conv2D(Conv2D_size, kernel_size=kernel_size_conv, activation=activation_func, input_shape=smpSize))
    model.add(MaxPooling2D(pool_size=pool_size_conv))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), 
             metrics=['accuracy'])
    return model


np.random.seed(1)
model = KerasClassifier(build_fn=create_model, batch_size=128, epochs=3)
param_grid = dict(kernel_size_conv=[(3, 3), (6, 6), (2, 2), (12, 12)], epochs=[3, 6], Conv2D_size=[32, 64],
                 activation_func=['relu', 'tanh'], pool_size_conv=[(2,2), (4, 4)])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(imgData, onehotData, verbose=1)

pd.DataFrame(grid_result.cv_results_)

print(grid_result.best_params_)

print(grid_result.best_score_)