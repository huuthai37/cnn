import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
import pickle
import random
import config

classes = 11


model = keras.applications.mobilenet.MobileNet(
    include_top=True,  
    weights='imagenet',
    dropout=0.5
)


# Modify network some last layer
x = Flatten()(model.layers[-4].output)
x = Dense(classes, activation='softmax', name='predictions')(x)

#Then create the corresponding model 
result_model = Model(inputs=model.input, outputs=x)
result_model.summary()
