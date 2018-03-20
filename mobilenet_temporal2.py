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

# train: python mobilenet_temporal2.py train 32 1 101
# test: python mobilenet_temporal2.py test 32 1 101
# retrain: python mobilenet_temporal2.py retrain 32 1 101 1
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
        out_file = '/home/oanhnt/thainh/data/database/train-opt2.pickle'
    else:
        out_file = '/home/oanhnt/thainh/data/database/test-opt2.pickle'
else:
    if train:
        out_file = '/mnt/smalldata/database/train-opt2.pickle'
    else:
        out_file = '/mnt/smalldata/database/test-opt2.pickle'

# MobileNet model
if train & (not retrain):
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        weights='imagenet'
    )
else:
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
    )

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

for i in range(2, len(layers)-3):
    x = layers[i](x)

x = Flatten()(x)
x = Dense(classes, activation='softmax', name='predictions')(x)

# Final touch
result_model = Model(inputs=input_new, outputs=x)

# Run
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/mobilenet_temporal2_{}e.h5'.format(old_epochs))
    else:
        result_model.get_layer('conv_new').set_weights(gd.convert_weights(layers[1].get_weights(), depth))

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)
    print('-'*40)
    print('MobileNet Sampled Optical stream only: Training')
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)
    
    for e in range(epochs):
        print('-'*40)
        print('Epoch', e+1)
        print('-'*40)

        random.shuffle(keys)
        result_model.fit_generator(gd.getTrainData(keys,batch_size,classes,3,train), verbose=1, max_queue_size=2, steps_per_epoch=len_samples/batch_size, epochs=1)
        result_model.save_weights('weights/mobilenet_temporal2_{}e.h5'.format(old_epochs+1+e))

else:
    result_model.load_weights('weights/mobilenet_temporal2_{}e.h5'.format(epochs))

    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)
    print('-'*40)
    print('MobileNet Sampled Optical stream only: Testing')
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)

    score = result_model.evaluate_generator(gd.getTrainData(keys,batch_size,classes,3,train), max_queue_size=3, steps=len_samples/batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])