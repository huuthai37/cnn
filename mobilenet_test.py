import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten, Input, ZeroPadding2D
import get_data as gd
from keras import optimizers
import pickle
import random
import numpy as np
import config

def convert_weights(weights):
    mat = weights[0]
    mat2 = np.empty([3,3,depth,32])
    for i in range(32):
        x=(mat[:,:,0,i] + mat[:,:,1,i] + mat[:,:,2,i])/3
        for j in range(depth):
            mat2[:,:,j,i] = x
    return [mat2]

# train: python mobilenet_temporal.py train 32 1 101
# test: python mobilenet_temporal.py test 32 1 101
# retrain: python mobilenet_temporal.py retrain 32 1 101 1
if sys.argv[1] == 'train':
    train = True
    retrain = False
    old_epochs = 0
elif sys.argv[1] == 'retrain':
    train = True
    retrain = True
    old_epochs = int(sys.argv[5])
else:
    train = False
    retrain = False

batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
classes = int(sys.argv[4])

depth = 20
input_shape = (224,224,depth)

server = config.server()
if server:
    if train:
        out_file = '/home/oanhnt/thainh/data/database/train-opt.pickle'
    else:
        out_file = '/home/oanhnt/thainh/data/database/test-opt.pickle'
else:
    if train:
        out_file = '/mnt/smalldata/database/train-opt.pickle'
    else:
        out_file = '/mnt/smalldata/database/test-opt.pickle'

# MobileNet model
if train & (not retrain):
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
else:
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
    )
model.summary()
# Disassemble layers
layers = [l for l in model.layers]

# Defining new convolutional layer.
input_new = Input(shape=input_shape)
x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(input_new)
x = Conv2D(filters=32, 
          kernel_size=(3, 3),
          padding='valid',
          use_bias=False,
		  strides=(2,2),
		  name='conv_new')(x)

for i in range(3, len(layers)-3):
    x = layers[i](x)

x = Flatten()(x)
x = Dense(classes, activation='softmax', name='predictions')(x)

# Final touch
result_model = Model(inputs=input_new, outputs=x)

result_model.summary()