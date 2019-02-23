import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')

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

model = create_model(64, (12,12), 'relu', (2,2))
model.fit(x=imgData, y=onehotData, batch_size=128,epochs=6)


testingImgData = test_data.values
testingImgData = testingImgData.reshape(21000, 28, 28, 1)
testingImgData = testingImgData / 255.0
testing_y_predict = model.predict(x=testingImgData, batch_size=128)
testing_y_predict = testing_y_predict.argmax(axis=-1)
test_data.insert(loc=0, column="label", 
    value=testing_y_predict)
test_data.to_csv("testingwlabels_cnn.csv", sep=',', 
    encoding='utf-8',index=False)


